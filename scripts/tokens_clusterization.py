
import torch
from datasets import Dataset, Audio
from transformers import Wav2Vec2Model, AutoProcessor

if __name__ == '__main__':

    sampling_rate = 16000

    ds = Dataset.load_from_disk("data/segments.dataset")
    ds = ds.cast_column('audio_path', Audio(sampling_rate=sampling_rate))

    processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")

    model = Wav2Vec2Model.from_pretrained("facebook/hubert-base-ls960", torch_dtype=torch.float16)

    all_hidden_states = []

    for segment_item in ds.select(range(300)):
        # Batch size 1
        input_values = processor(
            segment_item['audio_path']['array'],
            sampling_rate=sampling_rate,
            return_tensors="pt"
        ).input_values
        hidden_states = model(input_values.to(torch.float16)).last_hidden_state

        all_hidden_states.append({
            "hidden_states": hidden_states,
            "segment_num":   segment_item["segment_num"],
            "id":            segment_item["id"],
            "text":          segment_item["text"],
        })

    all_hidden_states
    breakpoint()