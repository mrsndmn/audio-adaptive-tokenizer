import os
import cProfile

import argparse

import pathlib
import torch
import torch.nn as nn

import logging
import evaluate

import datasets

from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM, HubertModel

import accelerate

from aat.model import TokenizedSpeechLM
from torch.optim.lr_scheduler import CyclicLR

from speechtokenizer import SpeechTokenizer

from aat.tokenizer import AdaptiveAudioAmplitudeTokenizer

from aat.training.config import TrainConfig, overfit_one_batch_train_config, full_unfreeze_train_config, AudioEncoderType
from aat.training.validate import val_loop
from aat.training.train import train_loop
from aat.training.dataloaders import build_dataloaders

from aat.lr_scheduler import WarmupLRScheduler

from aat.training.collate import TokenizedAudioWaveformCollator

from transformers.trainer import (
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def save_model(train_config: TrainConfig, model: TokenizedSpeechLM, path: pathlib.Path):
    path.mkdir(parents=True, exist_ok=True)
    logger.info(f"save model to {path}")

    model.save_pretrained(path)

    return


def train(
        model: TokenizedSpeechLM,
        tokenizer: transformers.AutoTokenizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        train_config: TrainConfig,
        device_placement=True,
        device=None,
        captioning_metrics=None,
        wer_compute=None,
        ):

    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.1,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=train_config.learning_rate)

    approximate_max_steps = len(train_dataloader.dataset) // train_dataloader.batch_size
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

        model = TokenizedSpeechLM.from_pretrained(audio_encoder, lm_decoder, projection_type=train_config.segment_projection, audio_encoder_type=train_config.audio_encoder_type,  model_id=from_pretrained)
    else:
        lm_decoder = build_lm_decoder(train_config, from_pretrained=train_config.lm_pretrained_model, device=device)
        model = TokenizedSpeechLM(audio_encoder, lm_decoder, projection_type=train_config.segment_projection, audio_encoder_type=train_config.audio_encoder_type)

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
    parser.add_argument('-p', '--profile', action='store_true', default=False)

    args = parser.parse_args()

        #  2. Capture a dictionary of hyperparameters
    # with open(args.config, 'r') as f:
    #     config_json_data = f.read()
    # train_config = TrainConfig.model_validate_json(config_json_data)

    if args.test_run:
        train_config = overfit_one_batch_train_config()
    else:
        train_config = full_unfreeze_train_config()

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
        )

    if args.profile:
        logger.info("Run training with profiling")
        with cProfile.Profile() as pr:
            run_training()

            profile_file_name = "train_profile.prof"
            logger.info(f"Save profile: {profile_file_name}")
            pr.dump_stats(profile_file_name)
    else:
        run_training()