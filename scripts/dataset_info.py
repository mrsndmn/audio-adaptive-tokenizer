from datasets import Dataset

ds = Dataset.load_from_disk("data/segments.dataset")

df = ds.to_pandas()

print("ID's Values Counts")
print(df['id'].value_counts())

print("\nTotal count of segments")
print(len(df))

print("\nDescribe count of segments")
print(df.segments_frames.map(lambda x: len(x)).describe())

breakpoint()