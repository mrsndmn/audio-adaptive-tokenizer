import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from speechtokenizer import SpeechTokenizer

from transformers import BertModel

class AudioEmbeddingsPooling(nn.Module):
    def __init__(self, bert_model):
        super().__init__()

        self.l_in = nn.Linear(576, bert_model.embeddings.word_embeddings.embedding_dim)
        self.bert = bert_model
        self.l_out = nn.Linear(bert_model.embeddings.word_embeddings.embedding_dim, 576)

        self.scale = nn.Parameter(torch.tensor([1.0]))

    def forward(self, inputs_embeds, encoder_attention_mask):

        projected_inputs = self.l_in(inputs_embeds)
        bert_outputs = self.bert(
            inputs_embeds=projected_inputs,
            encoder_attention_mask=encoder_attention_mask,
        )
        pooler_output = self.l_out(bert_outputs.pooler_output)
        pooler_output = F.normalize(pooler_output, dim=-1) * self.scale

        return pooler_output


class TokenizedSpeechLM(nn.Module):

    start_audio_token_id = 0
    end_audio_token_id = 1

    def __init__(self, audio_encoder, lm_decoder):
        super().__init__()

        self.audio_encoder = audio_encoder
        if isinstance(audio_encoder, SpeechTokenizer):
            self.embeddings_count = audio_encoder.quantizer.bins + 1
            self.speech_tokenizer_embeddings = nn.Embedding(self.embeddings_count, lm_decoder.config.hidden_size)

            self.projection = nn.Sequential(
                nn.Identity()
            )

            bert_model = BertModel.from_pretrained("bert-base-uncased")
            self.audio_embeddings_pooling = AudioEmbeddingsPooling(bert_model)

            self.speech_tokenizer_projection = nn.Linear(lm_decoder.config.hidden_size * 13, lm_decoder.config.hidden_size)
        else:
            # self.hubert = hubert # todo but required only for audio embeddings
            self.projection = nn.Sequential(
                # nn.Linear(768, 768*2),
                # nn.GELU(),
                nn.Linear(1024, lm_decoder.config.hidden_size),
                # nn.LayerNorm(lm_decoder.config.hidden_size),
            )

        self.audio_tokens_embeddings = nn.Embedding(2, lm_decoder.config.hidden_size)
        self.lm_decoder = lm_decoder

        return

    def reinitialize_weights(self, std=0.02):
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=std)
                nn.init.constant_(module.bias, 0)

        nn.init.normal_(self.audio_tokens_embeddings.weight, mean=0, std=std)

        if hasattr(self, 'speech_tokenizer_embeddings'):
            nn.init.normal_(self.speech_tokenizer_embeddings.weight, mean=0, std=std)

        if hasattr(self, 'audio_embeddings_pooling'):
            nn.init.normal_(self.audio_embeddings_pooling.l_in.weight, mean=0, std=std)
            nn.init.normal_(self.audio_embeddings_pooling.l_out.weight, mean=0, std=std)

        if hasattr(self, 'speech_tokenizer_projection'):
            nn.init.normal_(self.speech_tokenizer_projection.weight, mean=0, std=std)

        return

    def prepare_audio_embeddings(self, audio_embeds):
        audio_embeds = self.projection(audio_embeds)
        return audio_embeds

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, output_attentions=None):
        return self.lm_decoder.forward(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_attentions=None)

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

        torch.save(self.projection.state_dict(), os.path.join(save_directory, "projection.pt"))
        torch.save(self.audio_tokens_embeddings.state_dict(), os.path.join(save_directory, "audio_tokens_embeddings.pt"))

        if hasattr(self, 'speech_tokenizer_embeddings'):
            torch.save(self.speech_tokenizer_embeddings.state_dict(), os.path.join(save_directory, "speech_tokenizer_embeddings.pt"))

        if hasattr(self, 'audio_embeddings_pooling'):
            torch.save(self.audio_embeddings_pooling.state_dict(), os.path.join(save_directory, "audio_embeddings_pooling.pt"))

        if hasattr(self, 'speech_tokenizer_projection'):
            torch.save(self.speech_tokenizer_projection.state_dict(), os.path.join(save_directory, "speech_tokenizer_projection.pt"))

        self.lm_decoder.save_pretrained(save_directory)
        # self.config.save_pretrained(save_directory)

        return

    @classmethod
    def from_pretrained(cls, audio_encoder, lm_model, model_id: str):
        print("load TokenizedSpeechLM from", model_id)

        model = cls(audio_encoder, lm_model)

        projection_path = os.path.join(model_id, "projection.pt")

        projection_state = torch.load(projection_path, map_location=torch.device('cpu'))
        model.projection.load_state_dict(projection_state)

        audio_tokens_embeddings_path = os.path.join(model_id, "audio_tokens_embeddings.pt")
        audio_tokens_embeddings_state = torch.load(audio_tokens_embeddings_path, map_location=torch.device('cpu'))
        model.audio_tokens_embeddings.load_state_dict(audio_tokens_embeddings_state)

        if hasattr(model, 'speech_tokenizer_embeddings'):
            speech_tokenizer_embeddings_state_dict_path = os.path.join(model_id, "speech_tokenizer_embeddings.pt")
            speech_tokenizer_embeddings_state_dict = torch.load(speech_tokenizer_embeddings_state_dict_path, map_location=torch.device('cpu'))
            model.speech_tokenizer_embeddings.load_state_dict(speech_tokenizer_embeddings_state_dict)

        if hasattr(model, 'audio_embeddings_pooling'):
            audio_embeddings_pooling_state_dict_path = os.path.join(model_id, "audio_embeddings_pooling.pt")
            audio_embeddings_pooling_state_dict = torch.load(audio_embeddings_pooling_state_dict_path, map_location=torch.device('cpu'))
            model.speech_tokenizer_embeddings.load_state_dict(audio_embeddings_pooling_state_dict)

        if hasattr(model, 'speech_tokenizer_projection'):
            speech_tokenizer_projection_state_dict_path = os.path.join(model_id, "speech_tokenizer_projection.pt")
            speech_tokenizer_projection_state_dict = torch.load(speech_tokenizer_projection_state_dict_path, map_location=torch.device('cpu'))
            model.speech_tokenizer_projection.load_state_dict(speech_tokenizer_projection_state_dict)

        return model
