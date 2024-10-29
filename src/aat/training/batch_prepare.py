from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import accelerate

from aat.model import TokenizedSpeechLM
from aat.training.config import TrainConfig

def _inplace_audio_encode_batch_speechtokenizer(train_config: TrainConfig, model: TokenizedSpeechLM, batch, device=None):

    if hasattr(model, 'audio_embeddings_pooling'):
        # todo pad with audio_codes_with_cls_token
        # todo create seg
        segments_boarders_padded = batch['segments_boarders_padded']
        segments_boarders_attention_mask = batch['segments_boarders_attention_mask'] # [ bs, segments_count ]

        batch_size = segments_boarders_padded.shape[0]
        segments_count = segments_boarders_padded.shape[1]

        max_segment_waveform_frames = batch['segments_frames_len'].max()
        batched_segments = torch.zeros([batch_size, segments_count, max_segment_waveform_frames], device=device)

        waveforms_mask = torch.zeros_like(batched_segments)

        for batch_i in range(batch_size):
            prev_segment_boarder = 0
            for segment_i in range(segments_count):
                segment_boarder = segments_boarders_padded[batch_i, segment_i]
                if segment_i > 0 and segment_boarder == 0:
                    break
                segment_waveform = batch['audio_input_values'][batch_i, prev_segment_boarder:segment_boarder]
                batched_segments[batch_i, segment_i, :segment_waveform.shape[0]] = segment_waveform
                waveforms_mask[batch_i, segment_i, :segment_waveform.shape[0]] = 1
                prev_segment_boarder = segment_boarder

        # [ bs * segments_count, max_segment_waveform_frames ]
        batched_segments = batched_segments.flatten(0,1)

        audio_codes = model.audio_encoder.encode(
            batched_segments.unsqueeze(1),
            n_q=1,
        )
        # [ bs * segments_count, seq_len ]
        audio_codes = audio_codes.squeeze(0)
        # [ bs * segments_count, max_segment_waveform_frames ]
        waveforms_mask = waveforms_mask.flatten(0, 1)

        compression_factor = batched_segments.shape[-1] / audio_codes.shape[-1]
        compressed_seq_lengths = torch.round(waveforms_mask.sum(dim=-1) / compression_factor).to(torch.long)

        assert (compressed_seq_lengths != 0).any()

        codes_attention_mask = torch.arange(audio_codes.shape[-1], dtype=torch.long, device=device).unsqueeze(0).repeat(audio_codes.shape[0], 1)
        codes_attention_mask = (codes_attention_mask < compressed_seq_lengths.unsqueeze(1)).long()

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

        # [ bs * segments_count, embedding_dim ]
        # pooler_output

        # [ bs, segments_count, embedding_dim ]
        audio_hidden_states = pooler_output.unflatten(0, [batch_size, segments_count])
        # [ bs, segments_count ]
        codes_attention_mask = segments_boarders_attention_mask
    else:
        # [ 1, BS, seq_len ]
        audio_codes = model.audio_encoder.encode(
            batch['audio_input_values'].unsqueeze(1).to(device),
            n_q=1,
        )
        audio_codes = audio_codes.squeeze(0) # [ BS, seq_len ]

        # [ BS, seq_len, embedding_dim ]
        audio_hidden_states = model.speech_tokenizer_embeddings(audio_codes)

        compression_factor = batch['audio_input_values'].shape[-1] / audio_codes.shape[-1]
        compressed_seq_lengths = torch.round(batch['audio_attention_mask'].sum(dim=-1) / compression_factor).to(torch.long)

        assert (compressed_seq_lengths > 0).all()

        codes_attention_mask = torch.arange(audio_codes.shape[-1], dtype=torch.long).unsqueeze(0).repeat(audio_codes.shape[0], 1)
        codes_attention_mask = (codes_attention_mask < compressed_seq_lengths.unsqueeze(1)).long()

    batch['audio_embeds_last_hidden_state'] = audio_hidden_states
    batch['audio_embeds_attention_mask'] = codes_attention_mask

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

