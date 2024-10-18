import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenizedSpeechLM(nn.Module):

    start_audio_token_id = 0
    end_audio_token_id = 1

    def __init__(self, hubert, lm_decoder):
        super().__init__()

        # self.hubert = hubert # todo but required only for audio embeddings
        self.projection = nn.Sequential(
            nn.Linear(768, 768*2),
            nn.ReLU(),
            nn.Linear(768*2, 576),
            nn.LayerNorm(576),
        )

        self.audio_tokens_embeddings = nn.Embedding(2, lm_decoder.config.hidden_size)
        self.lm_decoder = lm_decoder

        return

    def prepare_audio_embeddings(self, audio_embeds):
        audio_embeddings = F.normalize(audio_embeddings, dim=-1)
        return self.lm_adapter(audio_embeds)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None):
        return self.lm_decoder.forward(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask)

    def encode_text(self, input_ids=None):
        return self.lm_decoder.model.embed_tokens(input_ids)

    def prepare_audio_inputs(self, input_ids=None, inputs_embeds=None, audio_embeds=None, attention_mask=None):

        if input_ids is not None:
            if inputs_embeds is not None:
                # logger.info("using inputs_embeds with input_ids the same time! inputs_embeds will be fist and then concatenated with input_ids")
                pass

            inputs_embeds = self.encode_text(input_ids)

        # [ bs, seq_len, llama_hidden_dim * self.config.modality_tokens ]
        if audio_embeds is None:
            raise Exception("no audio embeds")

        audio_embeds_projection = self.projection(audio_embeds)

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
            additional_attention_mask = torch.ones([bath_size, audio_tokens_seq_len], device=audio_embeds_projection.device)

            # мы можем это сделать тк атеншну все равно на какой позиции находятся токены,
            # главное, чтобы они были видны в атеншне
            attention_mask = torch.cat([additional_attention_mask, attention_mask], dim=1)
            assert attention_mask.shape[1] == inputs_embeds.shape[1], f"{attention_mask.shape[1]} == {inputs_embeds.shape[1]}"

        return {
            "inputs_embeds":  inputs_embeds,
            "attention_mask": attention_mask,
        }


    def save_pretrained(self, save_directory: str):
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        torch.save(self.projection.state_dict(), os.path.join(save_directory, "projection.pt"))
        torch.save(self.audio_tokens_embeddings.state_dict(), os.path.join(save_directory, "audio_tokens_embeddings.pt"))
        torch.save(self.lm_decoder.state_dict(), os.path.join(save_directory, "lm_decoder.pt"))

        # self.config.save_pretrained(save_directory)

        return

