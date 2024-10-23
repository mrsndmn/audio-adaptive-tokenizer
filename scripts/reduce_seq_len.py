from tqdm.auto import tqdm
from datasets import Dataset, load_dataset
import datasets

if __name__ == "__main__":
    ds = Dataset.load_from_disk("data/segments_tokenized_10_of_64.dataset/")
    ds = Dataset.load_from_disk("data/segments_tokenized_10_of_64.dataset/")

    dataset_files = [ f'libris/train-{i:05}-of-00064.parquet' for i in range(10) ] # 1 shard = 1 gb of data
    print("dataset_files", dataset_files)
    audio_dataset = load_dataset("nguyenvulebinh/asr-alignment", split=datasets.Split.TRAIN, data_files=dataset_files, streaming=True)
    audio_dataset = audio_dataset.remove_columns('audio')


    dataset_with_words_list = []
    for item, ad_item in zip(tqdm(ds), audio_dataset):

        dataset_with_words_list.append({
            **item,
            "words":      ad_item['words'],
            "word_end":   ad_item['word_end'],
            "word_start": ad_item['word_start'],
        })

    dataset_with_words_starts = Dataset.from_list(dataset_with_words_list)
    dataset_with_words_starts.save_to_disk("data/segments_tokenized_10_of_64_with_words_borders.dataset/")