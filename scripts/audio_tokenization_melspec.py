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

    audio_tokenizer = AdaptiveAudioAmplitudeTokenizer()

    audio_dataset = load_dataset("nguyenvulebinh/asr-alignment", 'libris')['train']
    audio_dataset = audio_dataset.cast_column('audio', datasets.Audio(sampling_rate=audio_tokenizer.sampling_rate))
    
    base_dir = "data/libris_melspectrograms"
    already_exists = set(os.listdir(base_dir))

    def process_item(item):
        item_id = item['id']
        if item_id in already_exists:
            return
        
        audio_waveform = item['audio']['array']
        audio_waveform_normed = (audio_waveform - audio_waveform.mean()) / (audio_waveform.std() + 1e-6)
        melspec = audio_tokenizer.get_melspec(audio_waveform_normed)
        torch.save(melspec, os.path.join(base_dir, item_id))

    audio_dataset.map(process_item)