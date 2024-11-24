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

from aslm.modeling_aslm import AslmModel, EfficientNetAudioEncdoerAdapter, EfficientNetAudioEncdoerConfig
from aslm.configuration_aslm import AslmConfig

from torch.optim.lr_scheduler import CyclicLR

from speechtokenizer import SpeechTokenizer

from aat.tokenizer import AdaptiveAudioAmplitudeTokenizer

from aat.training.config import TrainConfig, projection_training, finetuning_lm, overfit_one_batch_train_config
from aat.training.dataloaders import build_dataloaders

from aat.training.collate import NoSegmentationAudioWaveformCollator, TokenizedAudioWaveformCollator

from dataclasses import dataclass, field

from aat.training.trainer import AATTrainer, AATTrainerSegmentation, TrainingArguments, AudioEncoderType

from accelerate.tracking import filter_trackers

from aat.training.compute_metrics import ComputeMetrics

from aslm.configuration_aslm import AslmConfig, SegmentProjectionEnum

from aat.training.config import TrainConfig, SegmentationType

import wandb


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def train(
        model: AslmModel,
        tokenizer: transformers.AutoTokenizer,
        train_config: TrainConfig,
        training_args: TrainingArguments,
        ):

    audio_dataset = datasets.load_dataset("nguyenvulebinh/asr-alignment", 'libris')
    audio_dataset_val = audio_dataset['valid'].select(range(60))
    audio_dataset =  audio_dataset['train'] # .filter(lambda x: (x['audio']['array'].shape[-1] // x['audio']['sampling_rate']) < 18)
    if training_args.few_train_samples is not None:
        audio_dataset = audio_dataset.select(range(training_args.few_train_samples))
    audio_dataset = audio_dataset.shuffle(seed=42)
    
    early_stopping_callback = transformers.EarlyStoppingCallback(
        early_stopping_patience=20,
        early_stopping_threshold=0.01
    )
    
    if training_args.segmentation == SegmentationType.none:
        trainer = AATTrainer(
            model,
            training_args,
            processing_class=tokenizer,
            data_collator=NoSegmentationAudioWaveformCollator(train_config, tokenizer),
            train_dataset=audio_dataset,
            eval_dataset=audio_dataset_val,
            compute_metrics=ComputeMetrics(tokenizer),
            # callbacks=[ early_stopping_callback ],
        )
    elif training_args.segmentation == SegmentationType.uniform:
        
        # audio_encoder_embeddings_seq_len
        max_segment_frames = training_args.max_segment_frames
        max_segment_duration_milliseconds=(max_segment_frames * 1000 // train_config.sampling_rate)
        audio_tokenizer = AdaptiveAudioAmplitudeTokenizer(max_segment_duration_milliseconds=(max_segment_frames * 1000 // train_config.sampling_rate))
        
        trainer = AATTrainerSegmentation(
            model,
            training_args,
            processing_class=tokenizer,
            data_collator=TokenizedAudioWaveformCollator(training_args.segmentation, train_config, audio_tokenizer, tokenizer, uniform_segmentation_frames_per_segment=max_segment_frames),
            train_dataset=audio_dataset,
            eval_dataset=audio_dataset_val,
            compute_metrics=ComputeMetrics(tokenizer),
            # callbacks=[ early_stopping_callback ],
        )
    elif training_args.segmentation == SegmentationType.adaptive:
        
        max_segment_frames = training_args.max_segment_frames
        max_segment_duration_milliseconds=(max_segment_frames * 1000 // train_config.sampling_rate)
        audio_tokenizer = AdaptiveAudioAmplitudeTokenizer(max_segment_duration_milliseconds=max_segment_duration_milliseconds)

        trainer = AATTrainerSegmentation(
            model,
            training_args,
            processing_class=tokenizer,
            data_collator=TokenizedAudioWaveformCollator(training_args.segmentation, train_config, audio_tokenizer, tokenizer),
            train_dataset=audio_dataset,
            eval_dataset=audio_dataset_val,
            compute_metrics=ComputeMetrics(tokenizer),
            # callbacks=[ early_stopping_callback ],
        )


    else:
        raise ValueError("invalid segmentation value:", training_args.segmentation)


    trainer.accelerator.log_with = filter_trackers('wandb')
    trainer.accelerator.init_trackers(
        project_name="tokenized_speech_lm",
        config=train_config.model_dump()
    )

    trainer.train()

    breakpoint()
    
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

def build_lm_decoder(train_config: TrainConfig, training_args: TrainingArguments, from_pretrained=None, device=None):

    print("from_pretrained", from_pretrained)
    kwargs = dict()
    if train_config.lm_flash_attention:
        kwargs['torch_dtype'] = torch.float16
        kwargs['attn_implementation'] = "flash_attention_2"

    lm_decoder = LlamaForCausalLM.from_pretrained(from_pretrained, **kwargs)

    lm_decoder.to(device)

    return lm_decoder


def build_audio_encoder(train_config: TrainConfig, training_args: TrainingArguments, device=None):

    if training_args.audio_encoder_type == AudioEncoderType.hubert.value:
        kwargs = dict()
        if device is not None and 'cuda' in str(device):
            kwargs['torch_dtype'] = torch.float16
            kwargs['attn_implementation'] = "flash_attention_2"

        # model = HubertModel.from_pretrained("data/models/hubert_finetuned", mask_time_prob=0.0, **kwargs)
        model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft", mask_time_prob=0.0, **kwargs)
        # model.train()

        model = model.to(device)
    elif  training_args.audio_encoder_type == AudioEncoderType.efficient_net.value:
        from efficientnet_pytorch import EfficientNet
        eff_net = EfficientNet.from_pretrained('efficientnet-b0')
        
        config = EfficientNetAudioEncdoerConfig()
        model = EfficientNetAudioEncdoerAdapter(config, eff_net)
    else:
        raise ValueError(f"unknown audio_encoder_type: {training_args.audio_encoder_type}")

    return model


def build_model(train_config: TrainConfig, training_args: TrainingArguments, from_pretrained=None, device=None, audio_encoder_embeddings_seq_len=1, projection_type=SegmentProjectionEnum.linear):

    # lm_decoder = LlamaForCausalLM.from_pretrained("data/models/hearty-shadow-9/last")

    tokenizer = AutoTokenizer.from_pretrained(train_config.lm_pretrained_model)
    tokenizer.add_bos_token = True
    tokenizer.add_eos_token = True

    if 'qwen' in train_config.lm_pretrained_model.lower():
        tokenizer.bos_token_id = tokenizer.encode('<|im_start|>')[0]
        tokenizer.eos_token_id = tokenizer.encode('<|im_end|>')[0]

    audio_encoder = build_audio_encoder(train_config, training_args, device=device)

    if from_pretrained is not None:
        lm_decoder = build_lm_decoder(train_config, training_args, from_pretrained=train_config.lm_pretrained_model, device=device)

        model = AslmModel.from_pretrained(from_pretrained, audio_encoder, lm_decoder)
    else:
        lm_decoder = build_lm_decoder(train_config, training_args, from_pretrained=train_config.lm_pretrained_model, device=device)
        config = AslmConfig(
            audio_encoder_embeddings_seq_len=audio_encoder_embeddings_seq_len,
            projection_type=projection_type,
        )
        model = AslmModel(config, audio_encoder, lm_decoder)

        model.reinitialize_weights()

    model.to(device)

    if not training_args.train_audio_encoder:
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
    parser.add_argument('--projection_type', default='linear')

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
    elif args.profile:
        training_args.num_train_epochs = 1
        training_args.few_train_samples = 100
        training_args.per_device_train_batch_size = 10
        training_args.gradient_accumulation_steps = 1
        training_args.gradient_accumulation_steps = 1
        training_args.dataloader_num_workers = 0
        training_args.eval_steps = 100
        train_config = overfit_one_batch_train_config()
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
    audio_encoder_embeddings_seq_len = training_args.audio_encoder_embeddings_seq_len
    
    # uniform_segmentation_frames_per_segment
    
    training_args.output_dir = output_dir_base + f"_{audio_encoder_embeddings_seq_len}_{args.projection_type}_{training_args.segmentation}"

    model, tokenizer = build_model(
        train_config,
        training_args,
        device=device,
        from_pretrained=None,
        audio_encoder_embeddings_seq_len=audio_encoder_embeddings_seq_len,
        projection_type=args.projection_type,
    )

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
            run_training()

            profile_file_name = "train_profile.prof"
            logger.info(f"Save profile: {profile_file_name}")
            pr.dump_stats(profile_file_name)
    else:
        run_training()

