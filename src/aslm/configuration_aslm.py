# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Hubert model configuration"""

import functools
import operator

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from typing import Optional
from enum import Enum

logger = logging.get_logger(__name__)


class SegmentationType(str, Enum):
    none  = "none"
    uniform  = "uniform"
    adaptive = "adaptive"

class SegmentProjectionEnum(str, Enum):
    transformer_encoder  = "transformer_encoder"
    mean = "mean"
    linear = "linear"


class AslmConfig(PretrainedConfig):

    keys_to_ignore_at_inference = [ 'prefix_input_ids', ]

    r"""
    This is the configuration class to store the configuration of a [`AslmModel`].

    Args:
        vocab_size (`int`, *optional*, defaults to 32):
            Vocabulary size of the Hubert model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`HubertModel`]. Vocabulary size of the model. Defines the different
            tokens that can be represented by the *inputs_ids* passed to the forward method of [`HubertModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.


    Example:

    ```python
    >>> from transformers import HubertModel, HubertConfig

    >>> # Initializing a Hubert facebook/hubert-base-ls960 style configuration
    >>> configuration = HubertConfig()

    >>> # Initializing a model from the facebook/hubert-base-ls960 style configuration
    >>> model = HubertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "aslm"

    def __init__(
        self,
        projection_type: SegmentProjectionEnum = SegmentProjectionEnum.linear,
        audio_encoder_embeddings_seq_len: int = 1,
        uniform_segmentation_frames_per_segment: Optional[int] = None,
        max_segment_waveform_frames: Optional[int] = None,
        pad_token_id=-1,
        bos_token_id=0,
        eos_token_id=1,
        **kwargs,
    ):
        super().__init__(**kwargs, pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id)

        # Model
        self.projection_type = projection_type
        self.audio_encoder_embeddings_seq_len = audio_encoder_embeddings_seq_len

        # Segmentation
        self.uniform_segmentation_frames_per_segment = uniform_segmentation_frames_per_segment
        self.max_segment_waveform_frames = max_segment_waveform_frames

        return

