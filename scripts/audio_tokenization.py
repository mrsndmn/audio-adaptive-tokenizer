import datasets
from datasets import load_dataset

import os

from tqdm.auto import tqdm

import torchaudio
from transformers.audio_utils import spectrogram, mel_filter_bank, window_function

import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema

from datasets import Dataset, Audio
import statsmodels.api as sm

def find_amplitude_minimas(melspec):
    melspec_mean_amplitude = - 10 * melspec.mean(axis=0) # [ timesteps ]

    # amplitude smoothing
    melspec_mean_amplitude = sm.nonparametric.lowess(melspec_mean_amplitude, np.arange(len(melspec_mean_amplitude)), frac=0.008)
    melspec_mean_amplitude = melspec_mean_amplitude[:, 1]

    minimas = argrelextrema(melspec_mean_amplitude, np.greater)[0]

    return minimas

if __name__ == '__main__':
    n_fft = 400
    hop_length = 160
    feature_size = 80
    expected_sampling_rate = 16000

    audio_dataset = load_dataset("nguyenvulebinh/asr-alignment", split=datasets.Split.TRAIN, data_files=['libris/train-00001-of-00064.parquet'])
    audio_dataset.cast_column('audio', datasets.Audio(sampling_rate=expected_sampling_rate))

    mel_filters = mel_filter_bank(
        num_frequency_bins=1 + n_fft // 2,
        num_mel_filters=feature_size,
        min_frequency=0.0,
        max_frequency=8000.0,
        sampling_rate=expected_sampling_rate,
        norm="slaney",
        mel_scale="slaney",
    )

    window_fn = window_function(n_fft, "hann")

    minimal_segment_frames = int(expected_sampling_rate / 100) # 10мс
    maximum_segment_frames = int(expected_sampling_rate / 2)       # 1с

    processed_segments = []

    audio_segments_base_path = "./data/audio_segments"
    os.makedirs(audio_segments_base_path, exist_ok=True)

    for item in tqdm(audio_dataset):
        audio_waveform = item['audio']['array']

        item_melspec = spectrogram(
            audio_waveform,
            window_fn,
            frame_length=n_fft,
            hop_length=hop_length,
            power=2.0,
            mel_filters=mel_filters,
            log_mel="log10",
        )

        item_melspec_minimas = find_amplitude_minimas(item_melspec)
        item_waveform_minimas = item_melspec_minimas * hop_length
        if len(item_waveform_minimas) <= 1:
            # todo
            # return audio_waveform[0, :]
            raise NotImplementedError

        prev_minima = item_waveform_minimas[0]
        item_audio_segments = [ audio_waveform[0:prev_minima] ]
        # append the last frame for last segment
        item_waveform_minimas = item_waveform_minimas.tolist() + [ audio_waveform.shape[-1] ]

        for waveform_minima in item_waveform_minimas[1:]:
            segment_length_frames = waveform_minima - prev_minima

            if segment_length_frames < minimal_segment_frames:
                # filter out too small segments
                continue

            if segment_length_frames > maximum_segment_frames:
                # handle too big segments
                split_sizes = [ maximum_segment_frames ] * (segment_length_frames // maximum_segment_frames)
                split_sizes = np.cumsum(split_sizes)
                # print("item_waveform_minimas", item_waveform_minimas)
                # print('waveform_minima - prev_minima', waveform_minima, prev_minima, 'segment_length_frames', segment_length_frames, 'maximum_segment_frames', maximum_segment_frames)
                # print(split_sizes)
                splitted_waveform_segments = np.split(audio_waveform[prev_minima:waveform_minima], split_sizes)
                # print("splitted_waveform_segments", splitted_waveform_segments)
                item_audio_segments.extend(splitted_waveform_segments)
            else:
                item_audio_segments.append(audio_waveform[prev_minima:waveform_minima])

            prev_minima = waveform_minima

        assert len(item_audio_segments) < 200

        # print("item_audio_segments",  int(audio_waveform.shape[-1]/expected_sampling_rate), "s. segments count", len(item_audio_segments), [ f"{x.shape[-1]/expected_sampling_rate:.2f}" for x in item_audio_segments])
        for i, item_audio_segment in enumerate(item_audio_segments):
            segment_filename = os.path.join(audio_segments_base_path, item["id"] + "_" + str(i) + ".wav")
            torchaudio.save(segment_filename, torch.from_numpy(item_audio_segment).unsqueeze(0), sample_rate=expected_sampling_rate)

            processed_segments.append({
                "audio_path": segment_filename,
                "segment_num": i,
                "id": item["id"],
                "text": item["text"],
            })

    segmented_dataset = Dataset.from_list(processed_segments)
    segmented_dataset = segmented_dataset.cast_column("audio", Audio(sampling_rate=expected_sampling_rate))
    segmented_dataset.save_to_disk('data/segments.dataset')