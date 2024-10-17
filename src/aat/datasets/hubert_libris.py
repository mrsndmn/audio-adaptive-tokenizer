import torch
import numpy as np
from torch.utils.data import Dataset

class SegmentedHubertLibris(Dataset):
    def __init__(self, segments_dataset):
        self.segments_dataset = segments_dataset

    def __len__(self):
        return len(self.segments_dataset)

    def __getitem__(self, idx):
        item = self.segments_dataset[idx]

        audio_segments_embeddings = torch.load(item['segments_embeddings_path'])

        return {
            "text": item['text'],
            "audio_segments_embeddings": audio_segments_embeddings,
        }