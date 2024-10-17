import datasets
from datasets import load_dataset

dataset_iter = iter(load_dataset("nguyenvulebinh/asr-alignment", split=datasets.Split.TRAIN, data_files=['libris/train-00001-of-00064.parquet']))

dummy_dataset_sample = []
for i in range(1000):
    item = next(dataset_iter)

    dummy_dataset_sample.append(item)

print("dummy_dataset_sample", dummy_dataset_sample)