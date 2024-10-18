import datasets
from datasets import load_dataset
import shutil
import os

from transformers import Wav2Vec2Model, AutoProcessor

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

    processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")

    model = Wav2Vec2Model.from_pretrained("facebook/hubert-base-ls960", torch_dtype=torch.float16)


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

    minimal_segment_frames = int(expected_sampling_rate / 40) # 25мс
    maximum_segment_frames = int(expected_sampling_rate / 2)  # 0.5с

    processed_segments = []

    audio_segments_embeddings_base_path = "./data/audio_segments_embeddings"
    if os.path.exists(audio_segments_embeddings_base_path):
        shutil.rmtree(audio_segments_embeddings_base_path)
    os.makedirs(audio_segments_embeddings_base_path, exist_ok=True)

    for item in tqdm(audio_dataset.select(range(1500))):
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

        for i, waveform_minima in enumerate(item_waveform_minimas[1:]):
            segment_length_frames = waveform_minima - prev_minima

            if segment_length_frames < minimal_segment_frames:
                # filter out too small segments
                continue

            if segment_length_frames > maximum_segment_frames:
                # handle too big segments
                split_sizes = [ maximum_segment_frames ] * (segment_length_frames // maximum_segment_frames)
                split_sizes = np.cumsum(split_sizes)
                if segment_length_frames - split_sizes[-1] < minimal_segment_frames:
                    split_sizes[-1] = segment_length_frames - minimal_segment_frames
                # print(i, "item_waveform_minimas", item_waveform_minimas)
                # print(i, 'waveform_minima - prev_minima', waveform_minima, prev_minima, 'segment_length_frames', segment_length_frames, 'maximum_segment_frames', maximum_segment_frames)
                # print(i, split_sizes)
                splitted_waveform_segments = np.split(audio_waveform[prev_minima:waveform_minima], split_sizes)
                # print("splitted_waveform_segments", splitted_waveform_segments)
                item_audio_segments.extend(splitted_waveform_segments)
            else:
                item_audio_segments.append(audio_waveform[prev_minima:waveform_minima])

            prev_minima = waveform_minima

        assert len(item_audio_segments) < 200

        # print("item_audio_segments",  int(audio_waveform.shape[-1]/expected_sampling_rate), "s. segments count", len(item_audio_segments), [ f"{x.shape[-1]/expected_sampling_rate:.2f}" for x in item_audio_segments])
        segments_frames = []
        segments_embeddings = []

        segments_embeddings_file = os.path.join(audio_segments_embeddings_base_path, item['id'] + ".pt")
        for i, item_audio_segment in enumerate(item_audio_segments):
            if i == 0 and item_audio_segment.shape[-1] < minimal_segment_frames:
                # silence padding to fit minimul segment length
                item_audio_segment_padded = np.zeros([minimal_segment_frames])
                item_audio_segment_padded[-item_audio_segment.shape[-1]:] = item_audio_segment
                item_audio_segment = item_audio_segment_padded

            assert item_audio_segment.shape[-1] >= minimal_segment_frames, 'segment minimal length was not violated'
            segments_frames.append(item_audio_segment.shape[-1])

            input_values = processor(
                item_audio_segment,
                sampling_rate=expected_sampling_rate,
                return_tensors="pt"
            ).input_values
            hidden_states = model(input_values.to(torch.float16)).last_hidden_state

            segments_embeddings.append(hidden_states)

        torch.save(segments_embeddings, segments_embeddings_file)

        processed_segments.append({
            "id": item["id"],
            "audio_path": item['audio']['path'],
            "segments_embeddings_path": segments_embeddings_file,
            "segments_frames": segments_frames,
            "text": item["text"],
        })

    segmented_dataset = Dataset.from_list(processed_segments)
    segmented_dataset = segmented_dataset.cast_column("audio", Audio(sampling_rate=expected_sampling_rate))
    segmented_dataset.save_to_disk('data/segments.dataset')