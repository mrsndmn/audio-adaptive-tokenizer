
from datasets import concatenate_datasets, load_dataset, load_from_disk


ds1 = load_from_disk("data/libris_with_segments_shard_0.dataset/")
ds2 = load_from_disk("data/libris_with_segments_shard_1.dataset/")
ds3 = load_from_disk("data/libris_with_segments_shard_2.dataset/")
ds4 = load_from_disk("data/libris_with_segments_shard_3.dataset/")

merged = concatenate_datasets([ds1, ds2, ds3, ds4])

merged.save_to_disk("data/libris_with_segments_shard_1-4.dataset/")