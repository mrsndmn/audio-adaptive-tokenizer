from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import accelerate

from aat.model import TokenizedSpeechLM
from aat.training.config import TrainConfig, SegmentProjectionEnum

def _inplace_audio_encode_batch_speechtokenizer(train_config: TrainConfig, model: TokenizedSpeechLM, batch, device=None):

    segments_boarders_padded = batch['segments_boarders_padded']
    segments_boarders_attention_mask = batch['segments_boarders_attention_mask'].to(device) # [ bs, segments_count ]

    batch_size = segments_boarders_padded.shape[0]
    segments_count = segments_boarders_padded.shape[1]

    # [ bs * segments_count, max_segment_waveform_frames ]
    batched_segments = batch['batched_segments'].flatten(0,1).to(device)

    with torch.no_grad():
        audio_codes = model.audio_encoder.encode(
            batched_segments.unsqueeze(1),
            n_q=1,
        )

    # [ bs * segments_count, seq_len ]
    audio_codes = audio_codes.squeeze(0)

    # [ bs * segments_count, max_segment_waveform_frames ]
    segments_waveforms_mask = batch['segments_waveforms_mask'].flatten(0, 1).to(device)

    compression_factor = batched_segments.shape[-1] / audio_codes.shape[-1]
    compressed_seq_lengths = torch.round(segments_waveforms_mask.sum(dim=-1) / compression_factor).to(torch.long)

    assert (compressed_seq_lengths != 0).any()

    # [ bs * segments_count, seq_len ]
    codes_attention_mask = torch.arange(audio_codes.shape[-1], dtype=torch.long, device=device).unsqueeze(0).repeat(audio_codes.shape[0], 1)
    codes_attention_mask = (codes_attention_mask < compressed_seq_lengths.unsqueeze(1)).long()

    if train_config.segment_projection == SegmentProjectionEnum.transformer_encoder:
        cls_token = model.embeddings_count-1
        cls_token_tensor = torch.full([audio_codes.shape[0], 1], cls_token, device=device)
        audio_codes_with_cls = torch.cat([ cls_token_tensor, audio_codes ], dim=1)
        codes_attention_mask = torch.cat([ torch.ones(audio_codes.shape[0], 1, dtype=torch.long, device=device), codes_attention_mask ], dim=1)

        # [ bs * segments_count, seq_len + 1 (cls token), embedding_dim ]
        audio_hidden_states = model.speech_tokenizer_embeddings(audio_codes_with_cls)

        # [ bs * segments_count, seq_len + 1 (cls token), embedding_dim ]
        pooler_output = model.audio_embeddings_pooling.forward(
            inputs_embeds=audio_hidden_states,
            encoder_attention_mask=codes_attention_mask,
        )

        # [ bs, segments_count, embedding_dim ]
        audio_hidden_states = pooler_output.unflatten(0, [batch_size, segments_count])
    elif train_config.segment_projection == SegmentProjectionEnum.mean:
        # [ bs * segments_count, seq_len, embedding_dim ]
        audio_hidden_states = model.speech_tokenizer_embeddings(audio_codes)
        audio_hidden_states[~codes_attention_mask] = 0

        # [ bs * segments_count, embedding_dim ]
        sum_audio_hidden_states = audio_hidden_states.sum(dim=1)
        sum_codes_from_attention_mask = codes_attention_mask.sum(dim=-1, keepdim=True)
        sum_audio_hidden_states[sum_codes_from_attention_mask.squeeze(1) == 0] = 0

        audio_hidden_states = sum_audio_hidden_states / (sum_codes_from_attention_mask + 1e-9)

        # [ bs, segments_count, embedding_dim ]
        audio_hidden_states = audio_hidden_states.unflatten(0, [batch_size, segments_count])
    elif train_config.segment_projection == SegmentProjectionEnum.linear:
        # [ bs * segments_count, seq_len, embedding_dim ]
        audio_hidden_states = model.speech_tokenizer_embeddings(audio_codes)
        audio_hidden_states[~codes_attention_mask] = 0

        # seq_len is fixed due to each segment has fixed len
        # [ bs * segments_count, seq_len * embedding_dim ]
        audio_hidden_states = audio_hidden_states.flatten(1, 2)

        # [ bs * segments_count, embedding_dim ]
        audio_hidden_states = model.speech_tokenizer_projection(audio_hidden_states)
        # [ bs, segments_count, embedding_dim ]
        audio_hidden_states = audio_hidden_states.unflatten(0, [batch_size, segments_count])
    else:
        raise ValueError(f"unsupported segment_projection: {train_config.segment_projection}")

    batch['audio_embeds_last_hidden_state'] = audio_hidden_states
    batch['audio_embeds_attention_mask'] = segments_boarders_attention_mask

    return

def prepare_model_inputs_from_batch(train_config: TrainConfig, model: TokenizedSpeechLM, batch, device=None):

    _inplace_audio_encode_batch_speechtokenizer(train_config, model, batch, device=device)

    audio_embeds_last_hidden_state = batch['audio_embeds_last_hidden_state'].to(device)
    audio_embeds_attention_mask = batch['audio_embeds_attention_mask'].to(device)

    inputs_embeds = model.encode_text(batch['input_ids'].to(device))
    attention_mask = batch['attention_mask'].to(device)

    model_inputs_with_audio = model.prepare_audio_inputs(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        audio_embeds=audio_embeds_last_hidden_state,
        audio_embeds_attention_mask=audio_embeds_attention_mask,
    )

    return {
        "inputs_embeds":  model_inputs_with_audio["inputs_embeds"],
        "attention_mask": model_inputs_with_audio["attention_mask"],
    }

