from datasets import Dataset
from .hubert_libris import SegmentedHubertLibris

def test_dataset():
    ds = Dataset.load_from_disk("data/segments.dataset")

    segmented_ds = SegmentedHubertLibris(ds)
    segmented_ds_len = len(segmented_ds)
    count_segments = 0
    for x in segmented_ds:
        count_segments += len(x['audio_segments_embeddings'])

    assert segmented_ds_len == len(ds)
    assert count_segments > segmented_ds_len

