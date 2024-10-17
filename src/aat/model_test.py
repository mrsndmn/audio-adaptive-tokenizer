import torch
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutput
from .model import TokenizedSpeechLM

def test_tokenized_speech_lm():

    hubert = None
    lm_decoder = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")

    model = TokenizedSpeechLM(hubert, lm_decoder)

    seq_len = 10
    audio_embeddings = torch.rand([1, seq_len, 768])
    prepared_audio_embeds = model.prepare_audio_embeddings(audio_embeddings)

    outputs: CausalLMOutput = model.forward(inputs_embeds=prepared_audio_embeds)
    breakpoint()

    assert outputs.logits.shape == torch.Size([1, seq_len, lm_decoder.config.vocab_size])
