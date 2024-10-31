
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

class SegmentProjectionEnum(str, Enum):
    transformer_encoder  = "transformer_encoder"
    mean = "mean"
    linear = "linear"

class SegmentationType(str, Enum):
    uniform  = "uniform"
    adaptive = "adaptive"

class AudioEncoderType(str, Enum):
    hubert  = "hubert"
    speechTokenizer = "speechTokenizer"



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

    num_epochs: int = 500
    train_batch_size: int = 25
    val_batch_size: int = 1
    learning_rate: float = 3e-4
    # gradient_accumulation_steps = 2

    evaluate_every_epoch_mod: int = 10
    save_model_every_epoch_mod: int = 10

    no_validation: bool = False

    sampling_rate: int = 16000
    max_segment_waveform_frames: int = 4000

    # Model
    audio_encoder_type: AudioEncoderType
    audio_encoder_pretrained_model: str = "facebook/hubert-large-ls960-ft"
    lm_pretrained_model: str = "HuggingFaceTB/SmolLM-135M-Instruct"
    from_pretrained: Optional[str]

    optim_lm: bool = True
    unfreeze_lm_at_epoch: Optional[int]
    optim_audio_encoder: bool = False

    # Segmentation
    segmentation: SegmentationType
    uniform_segmentation_frames_per_segment: Optional[int]

    # Data
    few_train_samples: Optional[int] = 100
    few_val_samples: int = 1
    dataloader_num_workers: int = 5
    n_words: Optional[int] = None
    # dataloader_num_workers = 0

    train_dataset_path: str
    validation_dataset_path: str

    segment_projection: SegmentProjectionEnum
    segment_boarders_noize: bool = False

    @model_validator(mode='after')
    def validate_different_datasets(self):
        if self.train_dataset_path == self.validation_dataset_path:
            raise ValueError("Datasets must not be the same for validation and train")

        if self.segmentation == SegmentationType.uniform:
            if self.max_segment_waveform_frames != self.uniform_segmentation_frames_per_segment:
                raise ValueError("For uniform segmentation `uniform_segmentation_frames_per_segment` must be equal to `max_segment_waveform_frames`")


def overfit_one_batch_train_config():

    return TrainConfig(
        num_epochs = 1,
        train_batch_size = 10,
        val_batch_size = 1,
        learning_rate = 1e-4,
        # gradient_accumulation_steps = 2

        evaluate_every_epoch_mod = 10,
        save_model_every_epoch_mod = 10,

        no_validation = False,

        sampling_rate = 16000,
        max_segment_waveform_frames = 1600,

        # Model
        audio_encoder_type = AudioEncoderType.hubert,
        audio_encoder_pretrained_model = "facebook/hubert-large-ls960-ft",
        lm_pretrained_model = "HuggingFaceTB/SmolLM-135M-Instruct",
        from_pretrained = None,

        optim_lm = True,
        unfreeze_lm_at_epoch = None,
        optim_audio_encoder = False,

        segment_projection = SegmentProjectionEnum.linear,
        segmentation = SegmentationType.uniform,
        uniform_segmentation_frames_per_segment = 1600,
        segment_boarders_noize = True,

        # Data
        few_train_samples = 300,
        few_val_samples = 1,
        dataloader_num_workers = 0,
        n_words=5,

        train_dataset_path = "data/libris_with_segments_1_shard.dataset",
        validation_dataset_path = "data/libris_with_segments_valid.dataset",
    )


def full_unfreeze_train_config():

    return TrainConfig(
        num_epochs = 100,
        train_batch_size = 25,
        val_batch_size = 1,
        learning_rate = 1e-4,
        # gradient_accumulation_steps = 2

        evaluate_every_epoch_mod = 1,
        save_model_every_epoch_mod = 1,

        no_validation = False,

        sampling_rate = 16000,
        max_segment_waveform_frames = 1600,

        # Model
        audio_encoder_type = AudioEncoderType.hubert,
        audio_encoder_pretrained_model = "facebook/hubert-large-ls960-ft",
        lm_pretrained_model = "HuggingFaceTB/SmolLM-135M-Instruct",
        from_pretrained = 'data/models/creepy-wraith-137/last',

        optim_lm = True,
        unfreeze_lm_at_epoch = None,
        optim_audio_encoder = False,

        segment_projection = SegmentProjectionEnum.linear,
        segmentation = SegmentationType.uniform,
        uniform_segmentation_frames_per_segment = 1600,
        segment_boarders_noize = True,

        # Data
        few_train_samples = None,
        few_val_samples = 100,
        dataloader_num_workers = 10,
        n_words=50,

        train_dataset_path = "data/libris_with_segments_shard_1-4.dataset/",
        validation_dataset_path = "data/libris_with_segments_valid.dataset",
    )

