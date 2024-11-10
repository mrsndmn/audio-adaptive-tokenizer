import os
import cProfile

import argparse

import pathlib
import torch
import torch.nn as nn
from torch.amp import autocast

from typing import Optional

import logging
import evaluate

import datasets

from torch.utils.data import DataLoader
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM, HubertModel

import accelerate

from aat.model import AslmModel
from torch.optim.lr_scheduler import CyclicLR

from speechtokenizer import SpeechTokenizer

from aat.tokenizer import AdaptiveAudioAmplitudeTokenizer

from aat.training.config import TrainConfig, projection_training, finetuning_lm, AudioEncoderType
from aat.training.validate import val_loop
from aat.training.train import train_loop
from aat.training.dataloaders import build_dataloaders
from aat.training.optimizers import Adafactor

from aat.lr_scheduler import WarmupLRScheduler

from aat.training.collate import TokenizedAudioWaveformCollator

from transformers.trainer import (
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
)

from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def save_model(train_config: TrainConfig, model: AslmModel, path: pathlib.Path):
    path.mkdir(parents=True, exist_ok=True)
    logger.info(f"save model to {path}")

    model.save_pretrained(path)

    return


def train(
        model: AslmModel,
        tokenizer: transformers.AutoTokenizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        train_config: TrainConfig,
        device_placement=True,
        device=None,
        captioning_metrics=None,
        wer_compute=None,
        finetuning=False,
        ):


    approximate_max_steps = (len(train_dataloader.dataset) // train_dataloader.batch_size) * train_config.num_epochs
    logger.info(f"approximate_max_steps={approximate_max_steps}")
    optimizer_lr_scheduler = WarmupLRScheduler(optimizer, warmup_steps=300, max_steps=approximate_max_steps)
    # if train_config.max_lr > 0.0:
    #     optimizer_lr_scheduler = CyclicLR(optimizer, base_lr=train_config.learning_rate, max_lr=train_config.max_lr, step_size_up=train_config.step_size_up)

    # Иногда pad_token_id == eos_token_id,
    # но мы хотим, чтобы модель умела предсказывать eos_token_id
    # ignore_index=tokenizer.pad_token_id
    criterion = nn.CrossEntropyLoss()

    accelerator = accelerate.Accelerator(device_placement=device_placement, log_with='wandb')

    accelerator.init_trackers(
        project_name="tokenized_speech_lm",
        config=train_config.dict()
    )
    wandb_run = accelerator.get_tracker('wandb')

    if train_config.gradient_accumulation_steps is not None:
        accelerator.gradient_accumulation_steps = train_config.gradient_accumulation_steps

    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader)

    last_validation_wer=0.0

    for epoch in range(train_config.num_epochs):
        if train_config.unfreeze_lm_at_epoch is not None and epoch == train_config.unfreeze_lm_at_epoch:
            logger.info("Unfreeze lm decoder")
            unfreeze_model(model.lm_decoder)

        train_loop(accelerator, train_config, model, optimizer, optimizer_lr_scheduler, train_dataloader, epoch=epoch, criterion=criterion, last_validation_wer=last_validation_wer, device=device)

        if epoch % train_config.evaluate_every_epoch_mod == 0:
            validation_metrics = val_loop(train_config, model, tokenizer, val_dataloader, epoch=epoch, device=device, wer_compute=wer_compute, captioning_metrics=captioning_metrics)
            logger.info(f"validation metrics {validation_metrics}")
            last_validation_wer = validation_metrics.get('validation/wer', 0.0)

            accelerator.log(validation_metrics)

        if epoch % train_config.save_model_every_epoch_mod == 0:
            base_path_for_model = pathlib.Path(f"data/models/{wandb_run.run.name}/last/")
            save_model(train_config=train_config, model=model, path=base_path_for_model)

    base_path_for_model = pathlib.Path(f"data/models/{wandb_run.run.name}/last/")
    save_model(train_config=train_config, model=model, path=base_path_for_model)
    accelerator.end_training()

    return

def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False
    return

def unfreeze_model(model):
    for p in model.parameters():
        p.requires_grad = True
    return

def build_lm_decoder(train_config: TrainConfig, from_pretrained=None, device=None):

    print("from_pretrained", from_pretrained)
    kwargs = dict()
    if train_config.lm_flash_attention:
        kwargs['torch_dtype'] = torch.float16
        kwargs['attn_implementation'] = "flash_attention_2"

    lm_decoder = LlamaForCausalLM.from_pretrained(from_pretrained, **kwargs)

    lm_decoder.to(device)

    return lm_decoder


def build_audio_encoder(train_config: TrainConfig, device=None):

    if train_config.audio_encoder_type == AudioEncoderType.speechTokenizer:
        config_path = 'data/speechtokenizer/config.json'
        ckpt_path = 'data/speechtokenizer/ckpt.dev'
        model = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
        model.eval()
    else:
        kwargs = dict()
        if device is not None and 'cuda' in str(device):
            kwargs['torch_dtype'] = torch.float16
            kwargs['attn_implementation'] = "flash_attention_2"

        model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft", mask_time_prob=0.0, **kwargs)
        model.eval()

    model = model.to(device)

    return model


def build_model(train_config: TrainConfig, from_pretrained=None, device=None):

    # lm_decoder = LlamaForCausalLM.from_pretrained("data/models/hearty-shadow-9/last")

    tokenizer = AutoTokenizer.from_pretrained(train_config.lm_pretrained_model)
    tokenizer.add_bos_token = True
    tokenizer.add_eos_token = True

    if 'qwen' in train_config.lm_pretrained_model.lower():
        tokenizer.bos_token_id = tokenizer.encode('<|im_start|>')[0]
        tokenizer.eos_token_id = tokenizer.encode('<|im_end|>')[0]

    audio_encoder = build_audio_encoder(train_config, device=device)

    if from_pretrained is not None:
        lm_decoder = build_lm_decoder(train_config, from_pretrained=from_pretrained, device=device)

        model = AslmModel.from_pretrained(audio_encoder, lm_decoder, projection_type=train_config.segment_projection, audio_encoder_type=train_config.audio_encoder_type, hubert_embeddings_length_for_longest_audio_segment=train_config.hubert_embeddings_length_for_longest_audio_segment,  model_id=from_pretrained)
    else:
        lm_decoder = build_lm_decoder(train_config, from_pretrained=train_config.lm_pretrained_model, device=device)
        model = AslmModel(audio_encoder, lm_decoder, projection_type=train_config.segment_projection, audio_encoder_type=train_config.audio_encoder_type, hubert_embeddings_length_for_longest_audio_segment=train_config.hubert_embeddings_length_for_longest_audio_segment)

        model.reinitialize_weights()

    model.to(device)

    if not train_config.optim_audio_encoder:
        freeze_model(model.audio_encoder)

    if not train_config.optim_lm:
        freeze_model(model.lm_decoder)

    # unfreeze_model(model.audio_encoder)

    return model, tokenizer


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--config')
    parser.add_argument('-t', '--test-run', action='store_true', default=False)
    parser.add_argument('-f', '--finetune', action='store_true', default=False)
    parser.add_argument('-p', '--profile', action='store_true', default=False)

    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True

        #  2. Capture a dictionary of hyperparameters
    # with open(args.config, 'r') as f:
    #     config_json_data = f.read()
    # train_config = TrainConfig.model_validate_json(config_json_data)

    if args.finetune:
        train_config = finetuning_lm()
    else:
        train_config = projection_training()

    if args.test_run:
        train_config.few_train_samples = 100
        train_config.few_val_samples = 10
        train_config.train_batch_size = 10
        train_config.val_batch_size = 1
        train_config.num_epochs = 2

    device = train_config.nn_device
    device_placement = True
    logger.info(f"device {device}")

    logger.info("loading language model")

    model, tokenizer = build_model(train_config, from_pretrained=train_config.from_pretrained, device=device)

    logger.info("model was loaded")

    train_dataloader, val_dataloader = build_dataloaders(train_config, tokenizer)

    captioning_metrics = evaluate.combine(
        [
            evaluate.load("bleu", keep_in_memory=True),
            evaluate.load("rouge", keep_in_memory=True),
            evaluate.load("meteor", keep_in_memory=True),
        ]
    )
    wer_compute = evaluate.load("wer")

    # Initialise your wandb run, passing wandb parameters and any config information

    trainable_parameters_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_parameters_count = sum(p.numel() for p in model.parameters())
    logger.info(f"trainable model parameters: {trainable_parameters_count}")
    logger.info(f"total model parameters: {total_parameters_count}")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def run_training():
        train(
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            train_config=train_config,
            captioning_metrics=captioning_metrics,
            wer_compute=wer_compute,
            device=device,
            device_placement=True,
            finetuning=args.finetune,
        )

    if args.profile:
        logger.info("Run training with profiling")
        with cProfile.Profile() as pr:

            if train_config.optim_lm:
                run_training()
            else:
                with autocast(dtype=torch.bfloat16):
                    run_training()

            profile_file_name = "train_profile.prof"
            logger.info(f"Save profile: {profile_file_name}")
            pr.dump_stats(profile_file_name)
    else:
        run_training()

