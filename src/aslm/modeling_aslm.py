import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from aslm.configuration_aslm import AslmConfig, AudioEncoderType, SegmentProjectionEnum

from transformers import PreTrainedModel

class AudioEmbeddingsEncoderPooling(nn.Module):
    def __init__(self, embedding_dim=512, nhead=16):
        super().__init__()

        self.l_in = nn.Linear(576, embedding_dim)

        self.transformer_encoder = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            batch_first=True,
            norm_first=False
        )

        self.l_out = nn.Linear(embedding_dim, 576)

        self.scale = nn.Parameter(torch.tensor([1.0]))

    def forward(self, inputs_embeds, encoder_attention_mask):

        projected_inputs = self.l_in(inputs_embeds)
        transformer_encoder_outputs = self.transformer_encoder(
            src=projected_inputs,
            src_key_padding_mask=encoder_attention_mask.bool(),
        )

        # [bs * segments_count, 576]
        pooler_output = self.l_out(transformer_encoder_outputs[:, 0, :])
        pooler_output = F.normalize(pooler_output, dim=-1) * self.scale

        return pooler_output


class AslmModel(PreTrainedModel):

    config_class = AslmConfig

    def __init__(self, config: AslmConfig, audio_encoder, lm_decoder):
        super().__init__(config)

        self.audio_encoder = audio_encoder
        audio_encoder_hidden_size = audio_encoder.config.hidden_size

        assert config.audio_encoder_type == AudioEncoderType.hubert, 'only hubert audio encoder type is supported'

        if config.projection_type == SegmentProjectionEnum.transformer_encoder:
            self.audio_embeddings_pooling = AudioEmbeddingsEncoderPooling()
        elif config.projection_type == SegmentProjectionEnum.linear:
            linear_features = audio_encoder_hidden_size * config.hubert_embeddings_length_for_longest_audio_segment

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
        nn.init.normal_(self.audio_tokens_embeddings.weight, mean=0, std=std)

        if hasattr(self, 'audio_embeddings_pooling'):
            nn.init.normal_(self.audio_embeddings_pooling.l_in.weight, mean=0, std=std)
            nn.init.normal_(self.audio_embeddings_pooling.l_out.weight, mean=0, std=std)

        if hasattr(self, 'audio_encoder_projection'):
            if isinstance(self.audio_encoder_projection, nn.Linear):
                nn.init.normal_(self.audio_encoder_projection.weight, mean=0, std=std)
            if isinstance(self.audio_encoder_projection, nn.Sequential):
                for layer in self.audio_encoder_projection:
                    if isinstance(layer, nn.Linear):
                        nn.init.normal_(layer.weight, mean=0, std=std)

        return

    def encode_audio(self, waveform, waveforms_mask):
        """Encodes waveform to hidden dimension

        Args:
            waveform (Tensor) [ bs * segments_count, max_segment_waveform_frames ] : Segments waveforms
            waveforms_mask (Tensor) [ bs * segments_count, max_segment_waveform_frames ]: Padding mask for segments

        """

        if self.config.audio_encoder_type == AudioEncoderType.hubert:
            with torch.no_grad():
                ae_waveform = waveform
                if self.audio_encoder.dtype != ae_waveform.dtype:
                    ae_waveform = ae_waveform.to(self.audio_encoder.dtype)
                audio_hidden_states = self.audio_encoder(
                    input_values=ae_waveform,
                    attention_mask=waveforms_mask,
                ).last_hidden_state

                audio_hidden_states = audio_hidden_states.to(torch.float32)

            embeddings_attention_mask = self.audio_encoder._get_feature_vector_attention_mask(audio_hidden_states.shape[1], waveforms_mask)

        else:
            raise NotImplementedError

        assert not audio_hidden_states.isnan().any()

        assert embeddings_attention_mask.shape[1] == audio_hidden_states.shape[1]
        assert embeddings_attention_mask.shape[0] == audio_hidden_states.shape[0]

        # audio_hidden_states[~embeddings_attention_mask] = 0

        # audio_hidden_states ~ [ bs * segments_count, seq_len, embedding_dim ]
        # embeddings_attention_mask ~ [ bs * segments_count, seq_len ]
        return audio_hidden_states, embeddings_attention_mask


    def audio_embeddings_projection(self, audio_hidden_states, embeddings_attention_mask):
        if self.config.projection_type == SegmentProjectionEnum.transformer_encoder:
            raise NotImplementedError
        elif self.config.projection_type == SegmentProjectionEnum.mean:
            raise NotImplementedError
        elif self.config.projection_type == SegmentProjectionEnum.linear:
            seq_len = audio_hidden_states.shape[1]
            batch_size = audio_hidden_states.shape[0]
            cropped_seq_len = seq_len - (seq_len % self.config.hubert_embeddings_length_for_longest_audio_segment)
            redused_seq_len = cropped_seq_len // self.config.hubert_embeddings_length_for_longest_audio_segment
            audio_hidden_states_cropped = audio_hidden_states[:, :cropped_seq_len, :]
            assert audio_hidden_states_cropped.shape[1] > 0


            audio_hidden_states_cropped = audio_hidden_states_cropped.reshape(batch_size, redused_seq_len, -1)

            audio_hidden_states = self.audio_encoder_projection(audio_hidden_states_cropped)

            embeddings_attention_mask = embeddings_attention_mask[:, :cropped_seq_len].reshape(batch_size, redused_seq_len, -1).any(dim=-1)
        else:
            raise ValueError(f"unsupported projection_type: {self.config.projection_type}")

        audio_hidden_states = self.audio_encoder_dropout(audio_hidden_states)

        return audio_hidden_states, embeddings_attention_mask

    def prepare_audio_inputs(self, input_ids=None, inputs_embeds=None, audio_embeds=None, attention_mask=None, audio_embeds_attention_mask=None):

        if input_ids is not None:
            if inputs_embeds is not None:
                # logger.info("using inputs_embeds with input_ids the same time! inputs_embeds will be fist and then concatenated with input_ids")
                pass

            inputs_embeds = self.encode_text(input_ids)

        # [ bs, seq_len, llama_hidden_dim * self.config.modality_tokens ]
        if audio_embeds is None:
            raise Exception("no audio embeds")

        audio_embeds_projection, audio_embeds_attention_mask = self.audio_embeddings_projection(audio_hidden_states=audio_embeds, embeddings_attention_mask=audio_embeds_attention_mask)

        bath_size = audio_embeds_projection.shape[0]

        audio_start_end_tokens = torch.ones([bath_size, 2], device=audio_embeds_projection.device, dtype=torch.long)
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

        bath_size = audio_embeds_projection.shape[0]
        audio_tokens_seq_len = audio_embeds_projection.shape[1] + 2

        if attention_mask is not None:
            if audio_embeds_attention_mask is None:
                additional_attention_mask = torch.ones([bath_size, audio_tokens_seq_len], device=audio_embeds_projection.device)
            else:
                audio_tokens_labels_mask = torch.ones([bath_size, 1], device=audio_embeds_projection.device)
                additional_attention_mask = torch.cat([audio_tokens_labels_mask, audio_embeds_attention_mask, audio_tokens_labels_mask], dim=1)

            # мы можем это сделать тк атеншну все равно на какой позиции находятся токены,
            # главное, чтобы они были видны в атеншне
            attention_mask = torch.cat([additional_attention_mask, attention_mask], dim=1)
            assert attention_mask.shape[1] == inputs_embeds.shape[1], f"{attention_mask.shape[1]} == {inputs_embeds.shape[1]}"
        else:
            if audio_embeds_attention_mask is None:
                attention_mask = torch.ones([bath_size, audio_tokens_seq_len], device=audio_embeds_projection.device)
            else:
                audio_tokens_labels_mask = torch.ones([bath_size, 1], device=audio_embeds_projection.device)
                attention_mask = torch.cat([audio_tokens_labels_mask, audio_embeds_attention_mask, audio_tokens_labels_mask], dim=1)


        return {
            "inputs_embeds":  inputs_embeds,
            "attention_mask": attention_mask,
        }



    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, output_attentions=None):

        if inputs_embeds.dtype != self.lm_decoder.dtype:
            inputs_embeds = inputs_embeds.to(self.lm_decoder.dtype)

        assert inputs_embeds.shape[0] == attention_mask.shape[0]
        assert inputs_embeds.shape[1] == attention_mask.shape[1]

        return self.lm_decoder.forward(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_attentions=output_attentions)

    def encode_text(self, input_ids=None):
        return self.lm_decoder.model.embed_tokens(input_ids)

    def save_pretrained(self, save_directory: str):
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        torch.save(self.audio_tokens_embeddings.state_dict(), os.path.join(save_directory, "audio_tokens_embeddings.pt"))

        if hasattr(self, 'audio_embeddings_pooling'):
            torch.save(self.audio_embeddings_pooling.state_dict(), os.path.join(save_directory, "audio_embeddings_pooling.pt"))

        if hasattr(self, 'audio_encoder_projection'):
            torch.save(self.audio_encoder_projection.state_dict(), os.path.join(save_directory, "audio_encoder_projection.pt"))

        self.lm_decoder.save_pretrained(save_directory)
        # self.config.save_pretrained(save_directory)

        return

    @classmethod
    def from_pretrained(cls, audio_encoder, lm_model, projection_type, audio_encoder_type, hubert_embeddings_length_for_longest_audio_segment, model_id: str):
        print("load AslmModel from", model_id)

        model = cls(audio_encoder, lm_model, projection_type=projection_type, audio_encoder_type=audio_encoder_type, hubert_embeddings_length_for_longest_audio_segment=hubert_embeddings_length_for_longest_audio_segment)

        audio_tokens_embeddings_path = os.path.join(model_id, "audio_tokens_embeddings.pt")
        audio_tokens_embeddings_state = torch.load(audio_tokens_embeddings_path, map_location=torch.device('cpu'))
        model.audio_tokens_embeddings.load_state_dict(audio_tokens_embeddings_state)

        if hasattr(model, 'audio_embeddings_pooling'):
            audio_embeddings_pooling_state_dict_path = os.path.join(model_id, "audio_embeddings_pooling.pt")
            audio_embeddings_pooling_state_dict = torch.load(audio_embeddings_pooling_state_dict_path, map_location=torch.device('cpu'))
            model.audio_embeddings_pooling.load_state_dict(audio_embeddings_pooling_state_dict)

        if hasattr(model, 'audio_encoder_projection'):
            audio_encoder_projection_state_dict_path = os.path.join(model_id, "audio_encoder_projection.pt")
            audio_encoder_projection_state_dict = torch.load(audio_encoder_projection_state_dict_path, map_location=torch.device('cpu'))
            model.audio_encoder_projection.load_state_dict(audio_encoder_projection_state_dict)

        return model
