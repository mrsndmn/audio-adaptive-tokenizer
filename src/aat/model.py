
import torch
import torch.nn as nn


class TokenizedSpeechLM(nn.Module):

    def __init__(self, hubert, lm_decoder):
        super().__init__()

        # self.hubert = hubert # todo but required only for audio embeddings
        self.lm_adapter = nn.Sequential(
            nn.Linear(768, 768*2),
            nn.ReLU(),
            nn.Linear(768*2, 576),
        )

        self.lm_decoder = lm_decoder

        return

    def prepare_audio_embeddings(self, audio_embeds):
        return self.lm_adapter(audio_embeds)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None):

        return self.lm_decoder.forward(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask)