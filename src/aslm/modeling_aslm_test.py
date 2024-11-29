from tqdm.auto import tqdm
import torch
from transformers import AutoModelForCausalLM, HubertModel
from transformers.modeling_outputs import CausalLMOutput
from .modeling_aslm import AslmModel, AudioEmbeddingsEncoderPooling
from .configuration_aslm import AslmConfig, SegmentationType, SegmentProjectionEnum

import tempfile

def _prepare_model():
    hubert = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft", mask_time_prob=0.0)
    hubert.eval()

    lm_decoder = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
    lm_decoder.eval()

    config = AslmConfig(
        projection_type=SegmentProjectionEnum.linear,
        audio_encoder_embeddings_seq_len=1,
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


def test_audio_embeddings_encoder_pooling_backward_gradients():
    
    aeep = AudioEmbeddingsEncoderPooling(embedding_dim=128, hidden_dim=256, out_dim=128, nhead=8)
    
    batch_size, seq_len = 3, 7
    hidden_states = torch.rand([batch_size, seq_len, aeep.embedding_dim], requires_grad=True)
    encoder_attention_mask = torch.tensor([
        [ 1, 1, 1, 1, 1, 1, 1 ],
        [ 1, 1, 1, 1, 1, 0, 0 ],
        [ 1, 1, 1, 1, 1, 1, 0 ],
    ])

    outputs_no_ln = aeep.forward(hidden_states, encoder_attention_mask)
    rand_grads = torch.rand_like(outputs_no_ln)
    outputs_no_ln.backward(rand_grads.detach())

    assert hidden_states.grad is not None
    assert ((hidden_states.grad.sum(dim=-1) != 0) == encoder_attention_mask.bool()).all()
