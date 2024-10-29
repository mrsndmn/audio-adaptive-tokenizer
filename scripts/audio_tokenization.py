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

    # audio_dataset = load_dataset("nguyenvulebinh/asr-alignment", 'libris', split=datasets.Split.TRAIN)
    # dataset_files = [ f'libris/train-{i:05}-of-00064.parquet' for i in range(1) ]
    # dataset_files = [ f'libris/train-00063-of-00064.parquet' ]
    audio_dataset = datasets.load_dataset("nguyenvulebinh/asr-alignment", 'libris', streaming=True)
    audio_dataset = audio_dataset['train']
    audio_dataset.cast_column('audio', datasets.Audio(sampling_rate=expected_sampling_rate))

    audio_tokenizer = AdaptiveAudioAmplitudeTokenizer()

    libris_with_segments = []
    with torch.no_grad():
        for i, item in enumerate(tqdm(audio_dataset)):
            audio_waveform = item['audio']['array']

            awf_sr = AudioWaveform(audio_waveform, item['audio']['sampling_rate'])

            item_audio_segments = audio_tokenizer.tokenize(awf_sr)

            segment_frames = [ sf.waveform.shape[-1] for sf in item_audio_segments ]
            item['segment_frames'] = segment_frames
            libris_with_segments.append(item)

        segmented_dataset = Dataset.from_list(libris_with_segments)
        segmented_dataset.save_to_disk('data/libris_with_segments_full.dataset')
