import os

from typing import List
from transformers.audio_utils import spectrogram, mel_filter_bank, window_function

from scipy.signal import argrelextrema
import statsmodels.api as sm
import numpy as np
import torch

from aat.audio import AudioWaveform


class AdaptiveAudioAmplitudeTokenizer():
    def __init__(self,
                running_mean_points=12,
                min_segment_duration_milliseconds=125,
                max_segment_duration_milliseconds=1500,
                n_fft=400,
                hop_length=160,
                num_mel_filters=64,
                sampling_rate=16000,
                max_amplitude_for_minima=15,
            ):

        self.running_mean_points = running_mean_points
        self.max_amplitude_for_minima = max_amplitude_for_minima

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_mel_filters = num_mel_filters
        self.sampling_rate = sampling_rate

        self.min_segment_duration_milliseconds = min_segment_duration_milliseconds
        self.max_segment_duration_milliseconds = max_segment_duration_milliseconds

        self.min_segment_frames = self.milliseconds_to_frames(min_segment_duration_milliseconds)
        self.max_segment_frames = self.milliseconds_to_frames(max_segment_duration_milliseconds)


        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + self.n_fft // 2,
            num_mel_filters=num_mel_filters,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )

        self.window_fn = window_function(self.n_fft, "hann")

        return

    def find_amplitude_minimas(self, melspec: np.ndarray):
        """
        Finds local amplitude minnimas of the melspectrogram

        Args:
            melspec (np.ndarray of shape `(num_mel_filters, seq_len)`): Melspectrogram

        Returns:
            minimas (`np.array` of shape `(num_minimas)`): Array of minimas points of amplmitude at melspec
        """

        # Averaging over frequency dimension
        melspec_mean_amplitude = - 10 * melspec.mean(axis=0) # shape (seq_len)


        # Smooth amplitude over time
        def running_mean(x, N):
            cumsum = np.cumsum(x)
            return (cumsum[N:] - cumsum[:-N]) / float(N)

        melspec_mean_amplitude = running_mean(melspec_mean_amplitude, self.running_mean_points)

        # lowess is too slow
        # melspec_mean_amplitude = sm.nonparametric.lowess(melspec_mean_amplitude, np.arange(len(melspec_mean_amplitude)), frac=0.008)
        # melspec_mean_amplitude = melspec_mean_amplitude[:, 1] # shape (seq_len)

        # fp32 presision crutch
        def greater_eps(x1, x2):
            return x1 > x2 + 1e-5

        minimas = argrelextrema(melspec_mean_amplitude, greater_eps)[0] # shape (num_minimas)
        
        # filter out big amplitude minimas
        # that are corresponds to loud audio input
        # amplitude is negative np array
        minimas = minimas[ melspec_mean_amplitude[minimas] > self.max_amplitude_for_minima ]

        return minimas

    def milliseconds_to_frames(self, milliseconds: int) -> int:
        return int(milliseconds * self.sampling_rate / 1000)

    def left_pad_waveform_with_zeros(self, waveform):
        waveform_padded = np.zeros([self.min_segment_frames])
        waveform_padded[-waveform.shape[-1]:] = waveform
        return waveform_padded

    def right_pad_waveform_with_zeros(self, waveform):
        waveform_padded = np.zeros([self.min_segment_frames])
        waveform_padded[:waveform.shape[-1]] = waveform
        return waveform_padded

    def get_melspec(self, audio_waveform: np.ndarray) -> np.ndarray:

        melspec = spectrogram(
            audio_waveform,
            self.window_fn,
            frame_length=self.n_fft,
            hop_length=self.hop_length,
            power=2.0,
            mel_filters=self.mel_filters,
            log_mel="log10",
        )

        return melspec

    def pretokenize(self, audio_waveform: np.ndarray, melspec=None) -> List[int]:
        """Splits waveform to segments based on local minimas of amplidude

        Args:
            audio_waveform (`np.ndarray` of share `(n_frames)`): audio waveform

        Returns:
            List[int]: boundary frames indexes
        """
        if melspec is None:
            melspec = self.get_melspec(audio_waveform)

        melspec_minimas = self.find_amplitude_minimas(melspec)
        item_waveform_minimas = melspec_minimas * self.hop_length # move to waveform space

        # append the last frame as last segment end
        segments_boarders = item_waveform_minimas.tolist() + [ audio_waveform.shape[-1] ]

        return segments_boarders, melspec

    def process_segments_boarders(self, audio_waveform: np.ndarray, segments_boarders: np.ndarray) -> List[np.ndarray]:
        """Merge too small segments and split too big segments

        Args:
            audio_waveform (`np.ndarray` of shape `(n_frames)`): audio waveform
            segments_boarders (`np.ndarray` of shape `(n_segment_boarders)`): segments boarders from `self.pretokenize`

        Returns:
            List[np.ndarray]: list of result waveforms
        """

        waveform_segments: List[np.array] = []
        prev_minima = 0
        for waveform_minima in segments_boarders:
            segment_length_frames = waveform_minima - prev_minima

            if segment_length_frames < self.min_segment_frames:
                # ignore out too small segments
                continue

            if segment_length_frames > self.max_segment_frames:
                # handle too big segments
                split_sizes = [ self.max_segment_frames ] * (segment_length_frames // self.max_segment_frames)
                split_sizes = np.cumsum(split_sizes)
                last_frame_gap = segment_length_frames - split_sizes[-1]
                if last_frame_gap == 0:
                    split_sizes = split_sizes[:-1] # drop last empty segment
                elif last_frame_gap <  self.min_segment_frames:
                    split_sizes[-1] = segment_length_frames - self.min_segment_frames
                splitted_waveform_segments = np.split(audio_waveform[prev_minima:waveform_minima], split_sizes)
                waveform_segments.extend(splitted_waveform_segments)
            else:
                waveform_segments.append(audio_waveform[prev_minima:waveform_minima])

            prev_minima = waveform_minima

        if prev_minima != audio_waveform.shape[-1]:
            # right-pad the last segment with zeros
            last_segment = audio_waveform[prev_minima:]
            last_segment = self.right_pad_waveform_with_zeros(last_segment)
            waveform_segments.append(last_segment)

        return waveform_segments

    def tokenize(self, audio_waveform_sr: AudioWaveform, melspec=None) -> List[AudioWaveform]:

        audio_waveform_sr.assert_sampling_rate(self.sampling_rate)
        audio_waveform = audio_waveform_sr.waveform

        segments_boarders, melspec = self.pretokenize(audio_waveform, melspec=melspec)

        waveform_segments = self.process_segments_boarders(audio_waveform, segments_boarders)

        assert len(waveform_segments) < 300
        sum_frames = sum(x.shape[-1] for x in waveform_segments)
        assert sum_frames >= audio_waveform.shape[-1]

        audio_segments_sr: List[AudioWaveform] = [ AudioWaveform(wf, audio_waveform_sr.sampling_rate) for wf in waveform_segments ]

        return audio_segments_sr, melspec