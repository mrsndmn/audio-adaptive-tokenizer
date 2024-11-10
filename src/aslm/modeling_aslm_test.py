from tqdm.auto import tqdm
import torch
from transformers import AutoModelForCausalLM, HubertModel
from transformers.modeling_outputs import CausalLMOutput
from .modeling_aslm import AslmModel
from .configuration_aslm import AslmConfig, AudioEncoderType, SegmentationType, SegmentProjectionEnum

def _prepare_model():
    hubert = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft", mask_time_prob=0.0)
    hubert.eval()

    lm_decoder = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
    lm_decoder.eval()

    config = AslmConfig(
        audio_encoder_type=AudioEncoderType.hubert,
        projection_type=SegmentProjectionEnum.linear,
        hubert_embeddings_length_for_longest_audio_segment=1,
        segmentation=SegmentationType.none,
        uniform_segmentation_frames_per_segment=None,
        max_segment_waveform_frames=None,
    )
    model = AslmModel(config, hubert, lm_decoder)
    model = model.half()

    return model, lm_decoder

def test_tokenized_speech_lm():

    model, lm_decoder = _prepare_model()

    seq_len = 10
    audio_embeddings = torch.rand([1, seq_len, model.audio_encoder.config.hidden_size], dtype=torch.float16)
    attention_mask = torch.ones([1, seq_len])

    model_inputs = model.prepare_audio_inputs(audio_embeds=audio_embeddings, audio_embeds_attention_mask=attention_mask)
    outputs: CausalLMOutput = model.forward(inputs_embeds=model_inputs['inputs_embeds'], attention_mask=model_inputs['attention_mask'])

    expected_seq_len = seq_len + 2 # extra bos and eos tokens
    assert outputs.logits.shape == torch.Size([1, expected_seq_len, lm_decoder.config.vocab_size])

def test_tokenized_speech_lm_with_dataset():
    pass