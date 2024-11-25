import torch

import datasets

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


def test_collate_profile():
    
    audio_dataset = datasets.load_dataset("nguyenvulebinh/asr-alignment", 'libris')
    audio_dataset =  audio_dataset['train'] # .filter(lambda x: (x['audio']['array'].shape[-1] // x['audio']['sampling_rate']) < 18)

    train_config = overfit_one_batch_train_config()

    tokenizer = AutoTokenizer.from_pretrained(train_config.lm_pretrained_model)
    tokenizer.add_bos_token = True
    tokenizer.add_eos_token = True

    if 'qwen' in train_config.lm_pretrained_model.lower():
        tokenizer.bos_token_id = tokenizer.encode('<|im_start|>')[0]
        tokenizer.eos_token_id = tokenizer.encode('<|im_end|>')[0]

    max_segment_frames = 24000
    n_words = None
    max_segment_duration_milliseconds=(max_segment_frames * 1000 // 16000)
    audio_tokenizer = AdaptiveAudioAmplitudeTokenizer(
        max_segment_duration_milliseconds=max_segment_duration_milliseconds
    )

    collator = TokenizedAudioWaveformCollator("adaptive", train_config, audio_tokenizer, tokenizer, n_words=n_words)

    items = list(audio_dataset.select(range(20)))

    import cProfile
    with cProfile.Profile() as pr:
        for i in range(100):
            collator(items)

        profile_file_name = "test_collate_profile.prof"
        print(f"Save profile: {profile_file_name}")
        pr.dump_stats(profile_file_name)
