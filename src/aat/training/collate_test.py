import torch

from aat.training.collate import TokenizedAudioWaveformCollator
from aat.tokenizer import AdaptiveAudioAmplitudeTokenizer

import numpy as np
from transformers import AutoTokenizer

def test_collate():

    def build_text_tokenizer():
        text_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
        return text_tokenizer

    audio_tokenizer = AdaptiveAudioAmplitudeTokenizer()

    sampling_rate = 16000
    collator = TokenizedAudioWaveformCollator(audio_tokenizer, build_text_tokenizer, sampling_rate=sampling_rate)

    collator_items = [
        {
            "words": [ "Test", "example.", "Wow!" ],
            "word_end": [ 1.0, 5.0, 15.0 ],
            "audio": {
                "array": np.random.rand(15 * sampling_rate),
                "sampling_rate": sampling_rate,
            }
        },
        {
            "words": [ "The", "second", "example!", "Wow!" ],
            "word_end": [ 1.0, 5.0, 10.0, 15.0 ],
            "audio": {
                "array": np.random.rand(10 * sampling_rate), # different length for padding
                "sampling_rate": sampling_rate,
            }
        },
    ]
    collator_out = collator(collator_items)

    assert collator_out["audio_input_values"].shape == collator_out["audio_attention_mask"].shape
    assert collator_out["segments_boarders_padded"].shape == collator_out["segments_boarders_attention_mask"].shape
    assert collator_out["segments_max_frame_len"].shape == torch.Size([ len(collator_items) ])
