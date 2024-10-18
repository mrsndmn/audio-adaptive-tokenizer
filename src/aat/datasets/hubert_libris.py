import torch
import numpy as np
from datasets import Dataset

class SegmentedHubertLibris(torch.utils.data.Dataset):
    def __init__(self, segments_dataset):
        self.segments_dataset = segments_dataset

    def __len__(self):
        return len(self.segments_dataset)

    def __getitem__(self, idx):
        item = self.segments_dataset[idx]

        audio_segments_embeddings = torch.load(item['segments_embeddings_path'], weights_only=True)

        return {
            "text": item['text'],
            "audio_segments_embeddings": audio_segments_embeddings,
        }

    @classmethod
    def load_from_disk(klass, dataset_path):
        ds = Dataset.load_from_disk(dataset_path)
        return klass(ds)

