from tqdm.auto import tqdm
import torch
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutput
from .model import TokenizedSpeechLM

from .datasets.hubert_libris import SegmentedHubertLibris

def _prepare_model():
    hubert = None
    lm_decoder = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
    lm_decoder = lm_decoder.half()

    model = TokenizedSpeechLM(hubert, lm_decoder)
    model = model.half()

    return model, lm_decoder

def test_tokenized_speech_lm():

    model, lm_decoder = _prepare_model()

    seq_len = 10
    audio_embeddings = torch.rand([1, seq_len, 768], dtype=torch.float16)

    prepared_audio_embeds = model.prepare_audio_embeddings(audio_embeddings)
    outputs: CausalLMOutput = model.forward(inputs_embeds=prepared_audio_embeds)

    assert outputs.logits.shape == torch.Size([1, seq_len, lm_decoder.config.vocab_size])

def test_tokenized_speech_lm_with_dataset():

    model, lm_decoder = _prepare_model()

    dataset = SegmentedHubertLibris.load_from_disk("./data/segments.dataset")
    for dataset_item in tqdm(dataset):
        averaged_hubert_embeddings_list = [ x.mean(dim=1, keepdim=True) for x in dataset_item['audio_segments_embeddings'] ]
        averaged_hubert_embeddings_t = torch.cat(averaged_hubert_embeddings_list, dim=1) # [ 1, seq_length, 768 ]
        seq_len = averaged_hubert_embeddings_t.shape[1]
        assert seq_len < 200

        prepared_audio_embeds = model.prepare_audio_embeddings(averaged_hubert_embeddings_t) # [ 1, seq_length, 576 ]
        outputs: CausalLMOutput = model.forward(inputs_embeds=prepared_audio_embeds)

        assert outputs.logits.shape == torch.Size([1, seq_len, lm_decoder.config.vocab_size])
