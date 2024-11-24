
from typing import Optional

import torch
from pydantic import BaseModel, model_validator, computed_field, ConfigDict, Field
from enum import Enum

import os

class DeviceEnum(str, Enum):
    auto = "auto"
    mps  = "mps"
    cuda = "cuda"
    cpu  = "cpu"

from aslm.configuration_aslm import SegmentProjectionEnum, SegmentationType

class BaseExperiment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: DeviceEnum = Field(default=DeviceEnum.auto)

    @computed_field
    def nn_device(self) -> str:
        if self.device == 'auto':
            device = DeviceEnum.cpu
            if torch.backends.mps.is_available():
                device = DeviceEnum.mps
            elif torch.cuda.is_available():
                device = DeviceEnum.cuda
        else:
            device = self.device

        return device.value

class TrainConfig(BaseExperiment):
    # Training
    sampling_rate: int = 16000

    # Model TODO
    audio_encoder_pretrained_model: str = "facebook/hubert-large-ls960-ft"
    lm_pretrained_model: str = "HuggingFaceTB/SmolLM-135M-Instruct"
    lm_flash_attention: bool = False
    optim_lm: bool = True
    unfreeze_lm_at_epoch: Optional[int]

    # Data
    few_train_samples: Optional[int] = 100
    few_val_samples: int = 1
    n_words: Optional[int] = None
    add_prefix: bool = True

    not_segmented_dataset: bool = False

    train_dataset_path: str
    validation_dataset_path: str

    segment_projection: SegmentProjectionEnum

    @model_validator(mode='after')
    def validate_different_datasets(self):
        if self.train_dataset_path == self.validation_dataset_path:
            raise ValueError("Datasets must not be the same for validation and train")


def overfit_one_batch_train_config():

    return TrainConfig(
        sampling_rate = 16000,

        # Model
        audio_encoder_pretrained_model = "facebook/hubert-large-ls960-ft",
        lm_pretrained_model = "Qwen/Qwen1.5-1.8B",

        optim_lm = False,
        lm_flash_attention = True,
        unfreeze_lm_at_epoch = None,

        segment_projection = SegmentProjectionEnum.linear,

        # Data
        few_train_samples = 100,
        few_val_samples = 8,
        n_words=50,
        not_segmented_dataset = True,

        train_dataset_path = "data/libris_with_segments_full_processed.dataset/",
        validation_dataset_path = "data/libris_with_segments_valid.dataset/",
    )


def projection_training():

    return TrainConfig(

        sampling_rate = 16000,

        # Model
        audio_encoder_pretrained_model = "facebook/hubert-large-ls960-ft",
        lm_pretrained_model = "Qwen/Qwen1.5-1.8B",

        optim_lm = False,
        lm_flash_attention = False,
        unfreeze_lm_at_epoch = None,

        segment_projection = SegmentProjectionEnum.linear,

        # Data
        few_train_samples = None,
        few_val_samples = 100,
        n_words=50,
        not_segmented_dataset = True,

        train_dataset_path = "data/libris_with_segments_full_processed.dataset/",
        validation_dataset_path = "data/libris_with_segments_valid.dataset/",
    )

def finetuning_lm():

    return TrainConfig(
        sampling_rate = 16000,

        # Model
        audio_encoder_pretrained_model = "facebook/hubert-large-ls960-ft",
        lm_pretrained_model = "Qwen/Qwen1.5-1.8B",

        optim_lm = True,
        lm_flash_attention = False,
        unfreeze_lm_at_epoch = None,

        segment_projection = SegmentProjectionEnum.linear,
        # segmentation = SegmentationType.uniform,

        # Data
        few_train_samples = None,
        few_val_samples = 1000,
        n_words=50,
        not_segmented_dataset = True,

        train_dataset_path = "data/libris_with_segments_shard_1-4.dataset/",
        validation_dataset_path = "data/libris_with_segments_valid.dataset",
    )

