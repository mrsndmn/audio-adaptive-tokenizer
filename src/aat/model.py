import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from aat.training.config import SegmentProjectionEnum, AudioEncoderType

class AudioEmbeddingsPooling(nn.Module):
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


class TokenizedSpeechLM(nn.Module):

    start_audio_token_id = 0
    end_audio_token_id = 1

    def __init__(self, audio_encoder, lm_decoder, projection_type: SegmentProjectionEnum, audio_encoder_type: AudioEncoderType):
        super().__init__()

        self.audio_encoder_type = audio_encoder_type

        self.audio_encoder = audio_encoder

        self.audio_embeddings_scale = nn.Parameter(torch.tensor([1.0]))

        # Используется только для SpeechTokenizer, чтобы получить векторные представления из дискретных кодов
        if audio_encoder_type == AudioEncoderType.speechTokenizer:
            self.embeddings_count = audio_encoder.quantizer.bins + 1
            self.audio_encoder_embeddings = nn.Embedding(self.embeddings_count, lm_decoder.config.hidden_size)

        if projection_type == SegmentProjectionEnum.transformer_encoder:
            self.audio_embeddings_pooling = AudioEmbeddingsPooling()
        elif projection_type == SegmentProjectionEnum.linear:
            WAV_TOKENIZER_CODES_LENGTH_FOR_LONGEST_AUDIO_SEGMENT = 13
            WAV_TOKENIZER_CODES_LENGTH_FOR_LONGEST_AUDIO_SEGMENT = 5 # max_segment_waveform_frames == 1600
            # linear_features = lm_decoder.config.hidden_size * WAV_TOKENIZER_CODES_LENGTH_FOR_LONGEST_AUDIO_SEGMENT
            # HUBERT_EMBEDDINGS_LENGTH_FOR_LONGEST_AUDIO_SEGMENT = 24 # max_segment_waveform_frames == 1600
            HUBERT_EMBEDDINGS_LENGTH_FOR_LONGEST_AUDIO_SEGMENT = 1
            linear_features = 1024 * HUBERT_EMBEDDINGS_LENGTH_FOR_LONGEST_AUDIO_SEGMENT

            self.HUBERT_EMBEDDINGS_LENGTH_FOR_LONGEST_AUDIO_SEGMENT = HUBERT_EMBEDDINGS_LENGTH_FOR_LONGEST_AUDIO_SEGMENT

            # assert train_config.max_segment_waveform_frames == 4000, "WAV_TOKENIZER_CODES_LENGTH_FOR_LONGEST_AUDIO_SEGMENT relies on that"
            self.audio_encoder_projection = nn.Sequential(
                nn.Linear(linear_features, 4096),
                nn.ReLU(),
                nn.Linear(4096, lm_decoder.config.hidden_size),
            )
        elif projection_type == SegmentProjectionEnum.mean:
            # no special parameters
            linear_features = 1024 # hubert embedding dim
            self.audio_encoder_projection = nn.Linear(linear_features, lm_decoder.config.hidden_size)
            pass
        else:
            raise ValueError("Unhandled projection type:")

        self.audio_encoder_dropout = nn.Dropout(p=0.1)

        self.audio_tokens_embeddings = nn.Embedding(2, lm_decoder.config.hidden_size)
        self.lm_decoder = lm_decoder

        return

    def encode_audio(self, waveform, waveforms_mask):
        """Encodes waveform to hidden dimension

        Args:
            waveform (Tensor) [ bs * segments_count, max_segment_waveform_frames ] : Segments waveforms
            waveforms_mask (Tensor) [ bs * segments_count, max_segment_waveform_frames ]: Padding mask for segments

        """
        if self.audio_encoder_type == AudioEncoderType.speechTokenizer:
            with torch.no_grad():
                audio_codes = self.audio_encoder.encode(
                    waveform.unsqueeze(1),
                    n_q=1,
                )

            # [ bs * segments_count, seq_len ]
            audio_codes = audio_codes.squeeze(0)

            # [ bs * segments_count, max_segment_waveform_frames ]
            compression_factor = waveform.shape[-1] / audio_codes.shape[-1]
            compressed_seq_lengths = torch.round(waveforms_mask.sum(dim=-1) / compression_factor).to(torch.long)

            assert (compressed_seq_lengths != 0).any()

            # [ bs * segments_count, seq_len ]
            embeddings_attention_mask = torch.arange(audio_codes.shape[-1], dtype=torch.long, device=waveforms_mask.device).unsqueeze(0).repeat(audio_codes.shape[0], 1)
            embeddings_attention_mask = (embeddings_attention_mask < compressed_seq_lengths.unsqueeze(1)).long()

            audio_hidden_states = self.audio_encoder_embeddings(audio_codes)

        elif self.audio_encoder_type == AudioEncoderType.hubert:
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


    def reinitialize_weights(self, std=0.02):
        nn.init.normal_(self.audio_tokens_embeddings.weight, mean=0, std=std)

        if hasattr(self, 'audio_encoder_embeddings'):
            nn.init.normal_(self.audio_encoder_embeddings.weight, mean=0, std=std)

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

    def prepare_audio_embeddings(self, audio_embeds):
        # no op
        return audio_embeds

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, output_attentions=None):

        if inputs_embeds.dtype != self.lm_decoder.dtype:
            inputs_embeds = inputs_embeds.to(self.lm_decoder.dtype)

        assert inputs_embeds.shape[0] == attention_mask.shape[0]
        assert inputs_embeds.shape[1] == attention_mask.shape[1]

        return self.lm_decoder.forward(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_attentions=output_attentions)

    def encode_text(self, input_ids=None):
        return self.lm_decoder.model.embed_tokens(input_ids)

    def prepare_audio_inputs(self, input_ids=None, inputs_embeds=None, audio_embeds=None, attention_mask=None, audio_embeds_attention_mask=None):

        if input_ids is not None:
            if inputs_embeds is not None:
                # logger.info("using inputs_embeds with input_ids the same time! inputs_embeds will be fist and then concatenated with input_ids")
                pass

            inputs_embeds = self.encode_text(input_ids)

        # [ bs, seq_len, llama_hidden_dim * self.config.modality_tokens ]
        if audio_embeds is None:
            raise Exception("no audio embeds")

        audio_embeds_projection = self.prepare_audio_embeddings(audio_embeds)

        bath_size = audio_embeds_projection.shape[0]

        audio_start_end_tokens = torch.ones([bath_size, 2], device=audio_embeds_projection.device, dtype=torch.long)
        audio_start_end_tokens[:, 0] = audio_start_end_tokens[:, 0] * self.start_audio_token_id
        audio_start_end_tokens[:, 1] = audio_start_end_tokens[:, 1] * self.end_audio_token_id

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


    def save_pretrained(self, save_directory: str):
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        torch.save(self.audio_tokens_embeddings.state_dict(), os.path.join(save_directory, "audio_tokens_embeddings.pt"))
        torch.save(self.audio_embeddings_scale, os.path.join(save_directory, "audio_embeddings_scale.pt"))

        if hasattr(self, 'audio_encoder_embeddings'):
            torch.save(self.audio_encoder_embeddings.state_dict(), os.path.join(save_directory, "audio_encoder_embeddings.pt"))

        if hasattr(self, 'audio_embeddings_pooling'):
            torch.save(self.audio_embeddings_pooling.state_dict(), os.path.join(save_directory, "audio_embeddings_pooling.pt"))

        if hasattr(self, 'audio_encoder_projection'):
            torch.save(self.audio_encoder_projection.state_dict(), os.path.join(save_directory, "audio_encoder_projection.pt"))

        self.lm_decoder.save_pretrained(save_directory)
        # self.config.save_pretrained(save_directory)

        return

    @classmethod
    def from_pretrained(cls, audio_encoder, lm_model, projection_type, audio_encoder_type, model_id: str):
        print("load TokenizedSpeechLM from", model_id)

        model = cls(audio_encoder, lm_model, projection_type=projection_type, audio_encoder_type=audio_encoder_type)

        audio_tokens_embeddings_path = os.path.join(model_id, "audio_tokens_embeddings.pt")
        audio_tokens_embeddings_state = torch.load(audio_tokens_embeddings_path, map_location=torch.device('cpu'))
        model.audio_tokens_embeddings.load_state_dict(audio_tokens_embeddings_state)

        audio_embeddings_scale_path = os.path.join(model_id, "audio_embeddings_scale.pt")
        audio_embeddings_scale = torch.load(audio_embeddings_scale_path, map_location=torch.device('cpu'))
        model.audio_embeddings_scale = audio_embeddings_scale

        if hasattr(model, 'audio_encoder_embeddings'):
            audio_encoder_embeddings_state_dict_path = os.path.join(model_id, "audio_encoder_embeddings.pt")
            audio_encoder_embeddings_state_dict = torch.load(audio_encoder_embeddings_state_dict_path, map_location=torch.device('cpu'))
            model.audio_encoder_embeddings.load_state_dict(audio_encoder_embeddings_state_dict)

        if hasattr(model, 'audio_embeddings_pooling'):
            audio_embeddings_pooling_state_dict_path = os.path.join(model_id, "audio_embeddings_pooling.pt")
            audio_embeddings_pooling_state_dict = torch.load(audio_embeddings_pooling_state_dict_path, map_location=torch.device('cpu'))
            model.audio_embeddings_pooling.load_state_dict(audio_embeddings_pooling_state_dict)

        if hasattr(model, 'audio_encoder_projection'):
            audio_encoder_projection_state_dict_path = os.path.join(model_id, "audio_encoder_projection.pt")
            audio_encoder_projection_state_dict = torch.load(audio_encoder_projection_state_dict_path, map_location=torch.device('cpu'))
            model.audio_encoder_projection.load_state_dict(audio_encoder_projection_state_dict)

        return model
