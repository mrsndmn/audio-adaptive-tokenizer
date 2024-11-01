import datasets
from datasets import load_dataset
import shutil
import os

from transformers import AutoModel, AutoProcessor, HubertModel

from tqdm.auto import tqdm

import torch

from datasets import Dataset, Audio


from aat.tokenizer import AdaptiveAudioAmplitudeTokenizer
from aat.audio import AudioWaveform

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


if __name__ == '__main__':

    expected_sampling_rate = 16000

    audio_dataset = load_dataset("nguyenvulebinh/asr-alignment", 'libris')['train']
    audio_dataset = audio_dataset.cast_column('audio', datasets.Audio(sampling_rate=expected_sampling_rate))

    audio_tokenizer = AdaptiveAudioAmplitudeTokenizer()

    def process_item(item):
        audio_waveform = item['audio']['array']
        awf_sr = AudioWaveform(audio_waveform, item['audio']['sampling_rate'])
        item_audio_segments = audio_tokenizer.tokenize(awf_sr)
        segment_frames = [ sf.waveform.shape[-1] for sf in item_audio_segments ]
        return {"segment_frames": segment_frames}

    audio_dataset_processed = audio_dataset.map(process_item)
    audio_dataset_processed.save_to_disk(f'data/libris_with_segments_full_processed.dataset')

