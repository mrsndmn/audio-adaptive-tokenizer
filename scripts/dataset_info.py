from datasets import Dataset

ds = Dataset.load_from_disk("data/segments_tokenized_10_of_64.dataset")

df = ds.to_pandas()

print("ID's Values Counts")
print(df['id'].value_counts())

print("\nTotal count of segments")
print(len(df))

breakpoint()