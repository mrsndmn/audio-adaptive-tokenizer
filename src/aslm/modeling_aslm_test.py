from tqdm.auto import tqdm
import torch
from transformers import AutoModelForCausalLM, HubertModel
from transformers.modeling_outputs import CausalLMOutput
from .modeling_aslm import AslmModel
from .configuration_aslm import AslmConfig, AudioEncoderType, SegmentationType, SegmentProjectionEnum

import tempfile

def _prepare_model():
    hubert = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft", mask_time_prob=0.0)
    hubert.eval()

    lm_decoder = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
    lm_decoder.eval()

    config = AslmConfig(
        projection_type=SegmentProjectionEnum.linear,
        hubert_embeddings_length_for_longest_audio_segment=1,
        segmentation=SegmentationType.none,
        uniform_segmentation_frames_per_segment=None,
        max_segment_waveform_frames=None,
    )
    model = AslmModel(config, hubert, lm_decoder)

    return model, hubert, lm_decoder

def test_aslm_basic_forward():

    model, hubert, lm_decoder = _prepare_model()

    seq_len = 10
    audio_embeddings = torch.rand([1, seq_len, model.audio_encoder.config.hidden_size], dtype=model.dtype)
    attention_mask = torch.ones([1, seq_len])

    model_inputs = model.prepare_audio_inputs(audio_embeds=audio_embeddings, audio_embeds_attention_mask=attention_mask)
    outputs: CausalLMOutput = model.forward(inputs_embeds=model_inputs['inputs_embeds'], attention_mask=model_inputs['attention_mask'])

    expected_seq_len = seq_len + 2 # extra bos and eos tokens
    assert outputs.logits.shape == torch.Size([1, expected_seq_len, lm_decoder.config.vocab_size])

def test_aslm_save_load_pretrained():

    model, hubert, lm_decoder = _prepare_model()

    tempdir_name = tempfile.TemporaryDirectory()
    model.save_pretrained(tempdir_name.name, safe_serialization=True)

    model_restored = AslmModel.from_pretrained(tempdir_name.name, hubert, lm_decoder, use_safetensors=True)

    model_state_dict = model.state_dict()
    assert set(model_restored.state_dict().keys()) == set(model_state_dict.keys())

    for k, v in model_restored.state_dict().items():
        assert (model_state_dict[k] == v).all(), f'{k} mismatch'


# def test_batched_waveforms_with_no_values_in_attention_mask():