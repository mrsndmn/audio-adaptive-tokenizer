import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from aslm.configuration_aslm import AslmConfig, SegmentProjectionEnum

import safetensors
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast

class AudioEmbeddingsEncoderPooling(nn.Module):
    def __init__(self, embedding_dim=2048, hidden_dim=8192, out_dim=2048, nhead=32, num_layers=1, max_positions=64):
        super().__init__()

        self.l_in = nn.Linear(embedding_dim, hidden_dim)
        self.l_out = nn.Linear(hidden_dim, out_dim)
        # self.l_out = nn.Linear(embedding_dim, hidden_dim)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.num_layers = num_layers

        self.positional_embeddings = nn.Embedding(max_positions, hidden_dim)
        self.transformer_encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                batch_first=True,
                norm_first=True
            )
        ] * num_layers)

    def forward(self, inputs_embeds, encoder_attention_mask):
        hidden_states = self.l_in(inputs_embeds)
        
        hidden_states += self.positional_embeddings.weight[:hidden_states.shape[1], :]
        # hidden_states = self.layer_norm(hidden_states)

        for transformer_encoder_layer in self.transformer_encoder_layers:
            hidden_states_backup = hidden_states
            if hidden_states.isnan().any():
                print("found nans in hidden_states")
                breakpoint()

            hidden_states = transformer_encoder_layer(
                src=hidden_states,
                src_key_padding_mask=(~encoder_attention_mask.bool()),
            )
            if hidden_states.isnan().any():
                print("found nans in hidden_states")
                breakpoint()

        # [bs * segments_count, 1, hidden_dim]
        pooler_output = self.l_out(hidden_states[:, 0:1, :])
        # pooler_output = self.layer_norm(hidden_states[:, 0:1, :])

        return pooler_output

from dataclasses import dataclass

from efficientnet_pytorch.utils import Conv2dStaticSamePadding

@dataclass
class EfficientNetAudioEncdoerConfig:
    hidden_size: int = 1280

class EfficientNetAudioEncdoerAdapter(nn.Module):
    
    def __init__(self, config: EfficientNetAudioEncdoerConfig, efficient_net: nn.Module):
        super().__init__()
        
        self.config = config
        self.efficient_net = efficient_net
        self.efficient_net._fc = nn.Identity()
        self.efficient_net._dropout = nn.Identity()
        # self.efficient_net._avg_pooling = nn.Identity()
        
    def forward(
        self,
        input_values=None,
        attention_mask=None,
    ):
        efficient_net_output = self.efficient_net(input_values.repeat(1, 3, 1, 1))
        efficient_net_output = efficient_net_output.unsqueeze(1)
        # efficient_net_output reshape?
        return { "last_hidden_state": efficient_net_output }

    def _get_feature_vector_attention_mask(self, feature_seq_len, attention_mask=None, batch_size=None, device=None):

        return torch.ones([batch_size, feature_seq_len], device=device)

class AslmModel(PreTrainedModel):

    config_class = AslmConfig

    _keys_to_ignore_on_load_missing = [r"audio_encoder", r"lm_decoder"]

    def __init__(self, config: AslmConfig, audio_encoder, lm_decoder):
        super().__init__(config)

        self.audio_encoder = audio_encoder
        audio_encoder_hidden_size = audio_encoder.config.hidden_size

        if config.projection_type == SegmentProjectionEnum.transformer_encoder:
            self.audio_embeddings_pooling_cls_token = nn.Embedding(1, audio_encoder_hidden_size)
            # + 1 for bos token
            max_positions = config.audio_encoder_embeddings_seq_len + 1
            self.audio_embeddings_pooling = AudioEmbeddingsEncoderPooling(embedding_dim=audio_encoder_hidden_size, out_dim=lm_decoder.config.hidden_size, max_positions=max_positions)
            print("poolling params", sum(p.numel() for p in self.audio_embeddings_pooling.parameters()), "lm_decoder.config.hidden_size", lm_decoder.config.hidden_size)
        elif config.projection_type == SegmentProjectionEnum.linear:
            linear_features = audio_encoder_hidden_size * config.audio_encoder_embeddings_seq_len
            # self.audio_encoder_projection = nn.Sequential(
            #     nn.Linear(linear_features, lm_decoder.config.hidden_size),
            # )

            self.audio_encoder_projection = nn.Sequential(
                nn.Linear(linear_features, 4096),
                nn.ReLU(),
                nn.Linear(4096, lm_decoder.config.hidden_size),
            )
        elif config.projection_type == SegmentProjectionEnum.mean:
            # no special parameters
            linear_features = audio_encoder_hidden_size # hubert embedding dim
            self.audio_encoder_projection = nn.Linear(linear_features, lm_decoder.config.hidden_size)
        else:
            raise ValueError("Unhandled projection type:")

        self.audio_encoder_dropout = nn.Dropout(p=0.1)

        self.audio_tokens_embeddings = nn.Embedding(2, lm_decoder.config.hidden_size)
        self.lm_decoder = lm_decoder

        return


    def reinitialize_weights(self, std=0.02):
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

        attribute_names = [
            'audio_embeddings_pooling_cls_token',
            'audio_embeddings_pooling',
            'audio_encoder_projection',
            'audio_tokens_embeddings',
        ]
        for attribute_name in attribute_names:
            if not hasattr(self, attribute_name):
                continue
            attribute_value = self.__getattr__(attribute_name)
            attribute_value.apply(_init_weights)

        return

    def encode_audio(self, waveform, waveforms_mask=None, segments_boarders_attention_mask=None):
        """Encodes waveform to hidden dimension

        Args:
            waveform (Tensor) [ bs * segments_count, max_segment_waveform_frames ] : Segments waveforms
            waveforms_mask (Tensor) [ bs * segments_count, max_segment_waveform_frames ]: Padding mask for segments

        """

        # todo move this cases to projection class logic
        # [ bs * segments_count, max_segment_waveform_frames ]
        ae_waveform = waveform
        need_cast_to_fp32 = False
        if hasattr(self.audio_encoder, 'dtype') and self.audio_encoder.dtype != ae_waveform.dtype:
            need_cast_to_fp32 = True
            ae_waveform = ae_waveform.to(self.audio_encoder.dtype)

        # breakpoint()
        # audio_hidden_states ~ [ bs * segments_count, seq_len, embedding_dim ]
        audio_embeds = self.audio_encoder(
            input_values=ae_waveform,
            attention_mask=waveforms_mask,
        )['last_hidden_state']

        if need_cast_to_fp32:
            audio_embeds = audio_embeds.to(torch.float32)

        batch_size = audio_embeds.shape[0]
        new_seq_len = audio_embeds.shape[1]
        
        audio_embeds_attention_mask_hands = segments_boarders_attention_mask
        if audio_embeds_attention_mask_hands is None:
            audio_embeds_attention_mask_hands = waveforms_mask
            waveform_seq_len = waveforms_mask.shape[1]
            if waveform_seq_len % new_seq_len != 0:
                # extra_padding_length = new_seq_len - (waveform_seq_len % new_seq_len)
                # extra_padding = torch.zeros([batch_size, extra_padding_length], dtype=audio_embeds_attention_mask.dtype, device=audio_embeds_attention_mask.device)
                # audio_embeds_attention_mask = torch.cat([audio_embeds_attention_mask, extra_padding])
                audio_embeds_attention_mask_hands = audio_embeds_attention_mask_hands[:, :-(waveform_seq_len % new_seq_len)]

            audio_embeds_attention_mask_hands = audio_embeds_attention_mask_hands.reshape(batch_size, new_seq_len, -1).any(dim=-1).long()

        audio_embeds_negative_attention_mask = (audio_embeds_attention_mask_hands == 0)
        
        if waveforms_mask is not None:
            extra_kw_args = dict()
        else:
            extra_kw_args = {
                'batch_size': batch_size,
                'device': audio_embeds.device,
            }

        audio_embeds_attention_mask = self.audio_encoder._get_feature_vector_attention_mask(audio_embeds.shape[1], waveforms_mask, **extra_kw_args)
        audio_embeds_attention_mask[audio_embeds_negative_attention_mask] = 0

        assert not audio_embeds.isnan().any()

        assert audio_embeds_attention_mask.shape[1] == audio_embeds.shape[1]
        assert audio_embeds_attention_mask.shape[0] == audio_embeds.shape[0]

        # audio_hidden_states[~embeddings_attention_mask] = 0

        # audio_hidden_states ~ [ bs * segments_count, seq_len, embedding_dim ]
        # embeddings_attention_mask ~ [ bs * segments_count, seq_len ]
        return audio_embeds, audio_embeds_attention_mask


    def audio_embeddings_projection(self, audio_embeds, audio_embeds_attention_mask, segments_boarders_attention_mask=None):
        
        # audio_hidden_states ~ [ bs * segments_count, seq_len, embedding_dim ]
        # embeddings_attention_mask ~ [ bs * segments_count, seq_len ]

        # todo move this cases to projection class logic
        if self.config.projection_type == SegmentProjectionEnum.transformer_encoder:
            
            batch_size = audio_embeds.shape[0]
            
            cls_tokens_tensor = torch.zeros([ batch_size ], dtype=torch.long, device=audio_embeds.device).unsqueeze(1)
            cls_tokens_embeddings = self.audio_embeddings_pooling_cls_token(cls_tokens_tensor)
            audio_embeds_with_cls = torch.cat([cls_tokens_embeddings, audio_embeds], dim=1)
            
            cls_mask_value = torch.ones([ batch_size ], dtype=audio_embeds_attention_mask.dtype, device=audio_embeds_attention_mask.device).unsqueeze(1)
            audio_embeds_attention_mask_with_cls = torch.cat([cls_mask_value, audio_embeds_attention_mask], dim=-1)

            audio_embeds = self.audio_embeddings_pooling.forward(audio_embeds_with_cls, encoder_attention_mask=audio_embeds_attention_mask_with_cls)
            
            audio_embeds_attention_mask_reshaped = audio_embeds_attention_mask.reshape(batch_size, 1, -1).any(dim=-1)
            # if audio_embeds_attention_mask_reshaped.shape[0] > 1:
            #     assert audio_embeds_attention_mask_reshaped.sum() < audio_embeds_attention_mask_reshaped.numel()
            assert audio_embeds_attention_mask_reshaped.sum() > 0

            audio_embeds_attention_mask = audio_embeds_attention_mask_reshaped

        elif self.config.projection_type == SegmentProjectionEnum.mean:
            raise NotImplementedError
        elif self.config.projection_type == SegmentProjectionEnum.linear:
            audio_embeds[audio_embeds_attention_mask == 0] = 0

            seq_len = audio_embeds.shape[1]
            batch_size = audio_embeds.shape[0]
            
            # assert seq_len == self.config.audio_encoder_embeddings_seq_len, "is expected for segmented training"

            cropped_seq_len = seq_len - (seq_len % self.config.audio_encoder_embeddings_seq_len)
            redused_seq_len = cropped_seq_len // self.config.audio_encoder_embeddings_seq_len
            audio_hidden_states_cropped = audio_embeds[:, :cropped_seq_len, :]
            assert audio_hidden_states_cropped.shape[1] > 0
            

            audio_hidden_states_cropped = audio_hidden_states_cropped.reshape(batch_size, redused_seq_len, -1)

            audio_embeds = self.audio_encoder_projection(audio_hidden_states_cropped)

            audio_embeds_attention_mask_reshaped = audio_embeds_attention_mask[:, :cropped_seq_len].reshape(batch_size, redused_seq_len, -1).any(dim=-1)
            # if audio_embeds_attention_mask_reshaped.shape[0] > 1:
            #     assert audio_embeds_attention_mask_reshaped.sum() < audio_embeds_attention_mask_reshaped.numel()
            assert audio_embeds_attention_mask_reshaped.sum() > 0
            # if segments_boarders_attention_mask is not None:
            #     assert (segments_boarders_attention_mask.bool().flatten() == audio_embeds_attention_mask_reshaped.flatten()).all()

            audio_embeds_attention_mask = audio_embeds_attention_mask_reshaped
        else:
            raise ValueError(f"unsupported projection_type: {self.config.projection_type}")

        audio_embeds = self.audio_encoder_dropout(audio_embeds)
        
        assert audio_embeds.shape[0] == audio_embeds_attention_mask.shape[0]
        assert audio_embeds.shape[1] == audio_embeds_attention_mask.shape[1]

        return audio_embeds, audio_embeds_attention_mask

    def prepare_audio_inputs(self, input_ids=None, inputs_embeds=None, audio_embeds=None, attention_mask=None, audio_embeds_attention_mask=None, segments_count=None, segments_boarders_attention_mask=None):

        if input_ids is not None:
            if inputs_embeds is not None:
                # logger.info("using inputs_embeds with input_ids the same time! inputs_embeds will be fist and then concatenated with input_ids")
                pass

            inputs_embeds = self.encode_text(input_ids)

        # [ bs, seq_len, llama_hidden_dim * self.config.modality_tokens ]
        if audio_embeds is None:
            raise Exception("no audio embeds")

        audio_embeds_projection, audio_embeds_attention_mask = self.audio_embeddings_projection(audio_embeds=audio_embeds, audio_embeds_attention_mask=audio_embeds_attention_mask, segments_boarders_attention_mask=segments_boarders_attention_mask)
        # print("audio_embeds_attention_mask", audio_embeds_attention_mask.shape)

        batch_size = inputs_embeds.shape[0]

        # if segments_boarders_attention_mask is not None:
        #     assert (audio_embeds_attention_mask.flatten() == segments_boarders_attention_mask.flatten()).all()
        
        if segments_count is not None:
            # we have a del with a segmented model
            audio_embeds_projection = audio_embeds_projection.squeeze(1)
            audio_embeds_projection = audio_embeds_projection.unflatten(0, [batch_size, segments_count])
            audio_embeds_attention_mask = audio_embeds_attention_mask.squeeze(1)
            audio_embeds_attention_mask = audio_embeds_attention_mask.unflatten(0, [batch_size, segments_count])

        assert audio_embeds_projection.shape[0] == batch_size, "audio_embeds_projection batch size is equals to text embeddings batch size"

        audio_start_end_tokens = torch.ones([batch_size, 2], device=audio_embeds_projection.device, dtype=torch.long)
        audio_start_end_tokens[:, 0] = audio_start_end_tokens[:, 0] * self.config.bos_token_id
        audio_start_end_tokens[:, 1] = audio_start_end_tokens[:, 1] * self.config.eos_token_id

        audio_start_end_embeddings = self.audio_tokens_embeddings(audio_start_end_tokens)

        all_embeddings = [
            audio_start_end_embeddings[:, 0:1],
            audio_embeds_projection,
            audio_start_end_embeddings[:, 1:2],
        ]

        if inputs_embeds is not None:
            all_embeddings.append(inputs_embeds)

        inputs_embeds = torch.cat(all_embeddings, dim=1)

        batch_size = audio_embeds_projection.shape[0]
        audio_tokens_seq_len = audio_embeds_projection.shape[1] + 2

        if attention_mask is not None:
            if audio_embeds_attention_mask is None:
                additional_attention_mask = torch.ones([batch_size, audio_tokens_seq_len], device=audio_embeds_projection.device)
            else:
                audio_tokens_labels_mask = torch.ones([batch_size, 1], device=audio_embeds_projection.device)
                additional_attention_mask = torch.cat([audio_tokens_labels_mask, audio_embeds_attention_mask, audio_tokens_labels_mask], dim=1)

            # мы можем это сделать тк атеншну все равно на какой позиции находятся токены,
            # главное, чтобы они были видны в атеншне
            attention_mask = torch.cat([additional_attention_mask, attention_mask], dim=1)
            assert attention_mask.shape[1] == inputs_embeds.shape[1], f"{attention_mask.shape[1]} == {inputs_embeds.shape[1]}"
        else:
            if audio_embeds_attention_mask is None:
                attention_mask = torch.ones([batch_size, audio_tokens_seq_len], device=audio_embeds_projection.device)
            else:
                audio_tokens_labels_mask = torch.ones([batch_size, 1], device=audio_embeds_projection.device)
                attention_mask = torch.cat([audio_tokens_labels_mask, audio_embeds_attention_mask, audio_tokens_labels_mask], dim=1)

        return {
            "inputs_embeds":  inputs_embeds,
            "attention_mask": attention_mask,
            "audio_embeds": audio_embeds_projection,
            "audio_embeds_attention_mask": audio_embeds_attention_mask,
        }

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, output_attentions=None) -> BaseModelOutputWithPast:

        if inputs_embeds.dtype != self.lm_decoder.dtype:
            inputs_embeds = inputs_embeds.to(self.lm_decoder.dtype)

        assert inputs_embeds.shape[0] == attention_mask.shape[0]
        assert inputs_embeds.shape[1] == attention_mask.shape[1]

        return self.lm_decoder.forward(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_attentions=output_attentions)

    def encode_text(self, input_ids=None):
        return self.lm_decoder.model.embed_tokens(input_ids)

    def _prefixed_state_dict(self, key_prefix, state_dict):
        return { key_prefix + '.' + k: v for k, v in state_dict.items() }

    def save_pretrained(self, *args, **kwargs):
        save_audio_encoder = isinstance(self.audio_encoder, EfficientNetAudioEncdoerAdapter)
        state_dict_filtered = { k: v for k, v in self.state_dict().items() if not k.startswith('lm_decoder.') and (not k.startswith('audio_encoder.') or save_audio_encoder) }
        kwargs['state_dict'] = state_dict_filtered

        return super().save_pretrained(*args, **kwargs)
