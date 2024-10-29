
import datasets
from datasets import Dataset


dataset_files = [ f'libris/train-{i:05}-of-00064.parquet' for i in range(train_config.dataset_shards) ] # 1 shard = 1 gb of data

audio_dataset = datasets.load_dataset("nguyenvulebinh/asr-alignment", split=datasets.Split.TRAIN, data_files=dataset_files, streaming=False, shuffle=True)


