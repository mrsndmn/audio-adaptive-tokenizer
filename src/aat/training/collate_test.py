import torch

from aat.training.collate import TokenizedAudioWaveformCollator
from aat.tokenizer import AdaptiveAudioAmplitudeTokenizer
from aat.training.config import overfit_one_batch_train_config

import numpy as np
from transformers import AutoTokenizer

def test_collate():

    text_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")

    audio_tokenizer = AdaptiveAudioAmplitudeTokenizer()
    train_config = overfit_one_batch_train_config()

    sampling_rate = 16000
    collator = TokenizedAudioWaveformCollator(
        "uniform",
        train_config,
        audio_tokenizer,
        text_tokenizer,
        uniform_segmentation_frames_per_segment=4000,
    )

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

    assert collator_out["batched_segments"].shape == collator_out["segments_waveforms_mask"].shape
    assert collator_out["segments_boarders_padded"].shape == collator_out["segments_boarders_attention_mask"].shape
    assert collator_out["segments_max_frame_len"].shape == torch.Size([ len(collator_items) ])
    assert collator_out["batched_segments_melspectrograms"].shape == torch.Size([ len(collator_items), collator_out["segments_boarders_padded"].shape[1], 64, 9 ])
