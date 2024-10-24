import datasets
from datasets import load_dataset
import shutil
import os

from transformers import AutoModel, AutoProcessor, HubertModel

from tqdm.auto import tqdm

import torch

from datasets import Dataset, Audio


from aat.tokenizer import AdaptiveAudioAmplitudeTokenizer
from aat.audio import AudioWaveform

if __name__ == '__main__':

    processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")

    model = AutoModel.from_pretrained("facebook/hubert-large-ls960-ft")

    n_fft = 400
    hop_length = 160
    feature_size = 80
    expected_sampling_rate = 16000

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device", device)
    model.to(device)

    dataset_files = [ f'libris/train-{i:05}-of-00064.parquet' for i in range(1) ] # 1 shard = 1 gb of data
    print("dataset_files", dataset_files)
    audio_dataset = load_dataset("nguyenvulebinh/asr-alignment", split=datasets.Split.TRAIN, data_files=dataset_files, streaming=True)
    # audio_dataset = load_dataset("nguyenvulebinh/asr-alignment", 'libris', split=datasets.Split.TRAIN, streaming=True)
    audio_dataset.cast_column('audio', datasets.Audio(sampling_rate=expected_sampling_rate))

    audio_tokenizer = AdaptiveAudioAmplitudeTokenizer()

    processed_segments = []

    audio_segments_embeddings_base_path = "./data/audio_segments_embeddings_mean_tokenized_1"
    if os.path.exists(audio_segments_embeddings_base_path):
        shutil.rmtree(audio_segments_embeddings_base_path)
    os.makedirs(audio_segments_embeddings_base_path, exist_ok=True)

    with torch.no_grad():
        approximate_count_examples_in_one_shard = 4395
        for item in tqdm(audio_dataset, total=int(approximate_count_examples_in_one_shard * len(dataset_files))):
            audio_waveform = item['audio']['array']

            awf_sr = AudioWaveform(audio_waveform, item['audio']['sampling_rate'])

            item_audio_segments = audio_tokenizer.tokenize(awf_sr)

            segments_embeddings = []
            segments_frames = []

            segments_step = 100
            for i in range(0, len(item_audio_segments), segments_step):

                batch_waveforms = [ item_audio_segments[i+j].waveform for j in range(min(segments_step, len(item_audio_segments) - i)) ]

                segments_frames.extend([ seg.shape[-1] for seg in batch_waveforms ])

                input_params = processor(
                    batch_waveforms,
                    sampling_rate=expected_sampling_rate,
                    return_tensors="pt",
                    padding=True,
                )

                hidden_states = model(
                    input_values=input_params['input_values'].to(device),
                    attention_mask=input_params['attention_mask'].to(device),
                ).last_hidden_state

                padding_mask_for_hidden_states = model._get_feature_vector_attention_mask(hidden_states.shape[1], input_params['attention_mask'])
                padding_mask_for_hidden_states = padding_mask_for_hidden_states.to(device)

                hidden_states[~padding_mask_for_hidden_states] = 0.0

                # [ seq_len, 1024 ]
                pooled_output = hidden_states.sum(dim=1) / padding_mask_for_hidden_states.sum(dim=1, keepdim=True)

                # [ 1, seq_len, 1024 ]
                pooled_output = pooled_output.unsqueeze(0)
                # print("pooled_output", pooled_output.shape)
                # print("mean tokens", pooled_output.mean(dim=-1).mean())
                # print("mean norm", pooled_output.norm(2, dim=-1).mean())

                segments_embeddings.append(pooled_output)

            segments_embeddings_file = os.path.join(audio_segments_embeddings_base_path, item['id'] + ".pt")

            averaged_hubert_embeddings_t = torch.cat(segments_embeddings, dim=1) # [ 1, seq_length, 768 ]
            torch.save(averaged_hubert_embeddings_t, segments_embeddings_file)

            processed_segments.append({
                "id": item["id"],
                "audio_path": item['audio']['path'],
                "segments_embeddings_path": segments_embeddings_file,
                "segments_frames": segments_frames,
                "words": item['words'],
                "word_end": item['word_end'],
                "word_start": item['word_start'],
                "text": item["text"],
            })

        segmented_dataset = Dataset.from_list(processed_segments)
        segmented_dataset = segmented_dataset.cast_column("audio", Audio(sampling_rate=expected_sampling_rate))
        segmented_dataset.save_to_disk('data/segments_tokenized_1_of_64.dataset')