import datasets
from datasets import load_dataset
import shutil
import os

from transformers import Wav2Vec2Model, AutoProcessor

from tqdm.auto import tqdm

import torch

from datasets import Dataset, Audio


from aat.tokenizer import AdaptiveAudioAmplitudeTokenizer
from aat.audio import AudioWaveform

if __name__ == '__main__':

    processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")

    model = Wav2Vec2Model.from_pretrained("facebook/hubert-base-ls960", torch_dtype=torch.float16)


    n_fft = 400
    hop_length = 160
    feature_size = 80
    expected_sampling_rate = 16000

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device", device)
    model.to(device)

    audio_dataset = load_dataset("nguyenvulebinh/asr-alignment", split=datasets.Split.TRAIN, data_files=['libris/train-00001-of-00064.parquet'])
    audio_dataset.cast_column('audio', datasets.Audio(sampling_rate=expected_sampling_rate))

    audio_tokenizer = AdaptiveAudioAmplitudeTokenizer()

    processed_segments = []

    audio_segments_embeddings_base_path = "./data/audio_segments_embeddings"
    if os.path.exists(audio_segments_embeddings_base_path):
        shutil.rmtree(audio_segments_embeddings_base_path)
    os.makedirs(audio_segments_embeddings_base_path, exist_ok=True)

    for item in tqdm(audio_dataset.select(range(1500))):
        audio_waveform = item['audio']['array']

        awf_sr = AudioWaveform(audio_waveform, item['audio']['sampling_rate'])

        item_audio_segments = audio_tokenizer.tokenize(awf_sr)

        segments_embeddings = []
        segments_frames = []
        for i, audio_segment_wf_sr in enumerate(item_audio_segments):
            item_audio_segment = audio_segment_wf_sr.waveform

            segments_frames.append(item_audio_segment.shape[-1])

            input_values = processor(
                item_audio_segment,
                sampling_rate=expected_sampling_rate,
                return_tensors="pt"
            ).input_values
            hidden_states = model(input_values.to(torch.float16).to(device)).last_hidden_state

            segments_embeddings.append(hidden_states)

        segments_embeddings_file = os.path.join(audio_segments_embeddings_base_path, item['id'] + ".pt")
        torch.save(segments_embeddings, segments_embeddings_file)

        processed_segments.append({
            "id": item["id"],
            "audio_path": item['audio']['path'],
            "segments_embeddings_path": segments_embeddings_file,
            "segments_frames": segments_frames,
            "text": item["text"],
        })

    segmented_dataset = Dataset.from_list(processed_segments)
    segmented_dataset = segmented_dataset.cast_column("audio", Audio(sampling_rate=expected_sampling_rate))
    segmented_dataset.save_to_disk('data/segments.dataset')