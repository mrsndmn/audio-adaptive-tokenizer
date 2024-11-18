import os
import cProfile

import argparse

import pathlib
import torch
import torch.nn as nn
from torch.amp import autocast

from typing import Optional, List

import logging
import evaluate

import datasets

from torch.utils.data import DataLoader
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM, HubertModel

from aslm.modeling_aslm import AslmModel
from aslm.configuration_aslm import AslmConfig

from torch.optim.lr_scheduler import CyclicLR

from speechtokenizer import SpeechTokenizer


from aat.training.config import TrainConfig, projection_training, finetuning_lm, AudioEncoderType
from aat.training.dataloaders import build_dataloaders

from aat.training.collate import NoSegmentationAudioWaveformCollator

from dataclasses import dataclass, field

from aat.training.trainer import AATTrainer, TrainingArguments

from accelerate.tracking import filter_trackers

from aat.training.compute_metrics import ComputeMetrics

import wandb


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
        train_config: TrainConfig,
        training_args: TrainingArguments,
        ):


    # TODO LR Scheduler, optimizers

    # Иногда pad_token_id == eos_token_id,
    # но мы хотим, чтобы модель умела предсказывать eos_token_id
    # ignore_index=tokenizer.pad_token_id

    audio_dataset = datasets.load_dataset("nguyenvulebinh/asr-alignment", 'libris')
    audio_dataset_val = audio_dataset['valid'].select(range(60))
    audio_dataset =  audio_dataset['train']
    audio_dataset = audio_dataset.shuffle(seed=42)
    
    early_stopping_callback = transformers.EarlyStoppingCallback(
        early_stopping_patience=5,
        early_stopping_threshold=0.005
    )
    
    trainer = AATTrainer(
        model,
        training_args,
        processing_class=tokenizer,
        data_collator=NoSegmentationAudioWaveformCollator(train_config, tokenizer),
        train_dataset=audio_dataset,
        eval_dataset=audio_dataset_val,
        compute_metrics=ComputeMetrics(tokenizer),
        callbacks=[ early_stopping_callback ],
    )

    trainer.accelerator.log_with = filter_trackers('wandb')
    trainer.accelerator.init_trackers(
        project_name="tokenized_speech_lm",
        config=train_config.model_dump()
    )

    trainer.train()
    
    trainer.accelerator.end_training()

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

    kwargs = dict()
    if device is not None and 'cuda' in str(device):
        kwargs['torch_dtype'] = torch.float16
        kwargs['attn_implementation'] = "flash_attention_2"

    model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft", mask_time_prob=0.0, **kwargs)
    model.eval()

    model = model.to(device)

    return model


def build_model(train_config: TrainConfig, from_pretrained=None, device=None, hubert_embeddings_length_for_longest_audio_segment=7):

    # lm_decoder = LlamaForCausalLM.from_pretrained("data/models/hearty-shadow-9/last")

    tokenizer = AutoTokenizer.from_pretrained(train_config.lm_pretrained_model)
    tokenizer.add_bos_token = True
    tokenizer.add_eos_token = True

    if 'qwen' in train_config.lm_pretrained_model.lower():
        tokenizer.bos_token_id = tokenizer.encode('<|im_start|>')[0]
        tokenizer.eos_token_id = tokenizer.encode('<|im_end|>')[0]

    audio_encoder = build_audio_encoder(train_config, device=device)

    if from_pretrained is not None:
        lm_decoder = build_lm_decoder(train_config, from_pretrained=train_config.lm_pretrained_model, device=device)

        model = AslmModel.from_pretrained(from_pretrained, audio_encoder, lm_decoder)
    else:
        lm_decoder = build_lm_decoder(train_config, from_pretrained=train_config.lm_pretrained_model, device=device)
        config = AslmConfig(hubert_embeddings_length_for_longest_audio_segment=hubert_embeddings_length_for_longest_audio_segment)
        model = AslmModel(config, audio_encoder, lm_decoder)

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

    args, remainig_args = parser.parse_known_args()

    hf_parser = transformers.HfArgumentParser(TrainingArguments)
    (training_args,) = hf_parser.parse_args_into_dataclasses(args=remainig_args)

    torch.backends.cuda.matmul.allow_tf32 = True

        #  2. Capture a dictionary of hyperparameters
    # with open(args.config, 'r') as f:
    #     config_json_data = f.read()
    # train_config = TrainConfig.model_validate_json(config_json_data)

    if args.finetune:
        train_config = finetuning_lm()
        training_args.num_train_epochs = 1
        training_args.per_device_train_batch_size = 20
        training_args.gradient_accumulation_steps = 5
        training_args.eval_steps = 300
    else:
        train_config = projection_training()

    if args.test_run:
        train_config.few_train_samples = 100
        train_config.few_val_samples = 10
        train_config.train_batch_size = 10
        train_config.val_batch_size = 1
        train_config.num_epochs = 2

    device = train_config.nn_device
    logger.info(f"device {device}")

    logger.info("loading language model")
    
    output_dir_base = training_args.output_dir
    
    for hubert_embeddings_length_for_longest_audio_segment in range(11, 20, 2):
        training_args.output_dir = output_dir_base + f"_{hubert_embeddings_length_for_longest_audio_segment}"

        model, tokenizer = build_model(train_config, device=device, from_pretrained=None, hubert_embeddings_length_for_longest_audio_segment=hubert_embeddings_length_for_longest_audio_segment)

        logger.info("model was loaded")

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
                train_config=train_config,
                training_args=training_args,
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

