import os
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import Trainer
from transformers.trainer import (
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional, Union, Any, Union, Dict, Tuple

from torch.optim import Adam, AdamW

from aat.training.config import TrainConfig

from aslm.modeling_aslm import AslmModel
from aslm.configuration_aslm import SegmentationType
from transformers.modeling_outputs import BaseModelOutputWithPast

from dataclasses import dataclass, field
import transformers

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = field(default="data/models")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    dataloader_drop_last: bool = field(default=True)
    dataloader_num_workers: int = field(default=10)
    per_device_train_batch_size: int = field(default=50)
    gradient_accumulation_steps: int = field(default=1)

class AATTrainer(Trainer):

    args: TrainingArguments
    model: AslmModel

    def create_optimizer(self):
        opt_model = self.model
        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        if self.optimizer is None:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.1,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer


    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """

        # assert self.args.segmentation == SegmentationType.none

        # move to device
        # [ bs, max_segment_waveform_frames ]
        batched_waveforms = inputs['waveforms'].to(device=self.args.device)
        batched_waveforms_attention_mask = inputs['waveforms_attention_mask'].to(device=self.args.device)

        # audio_hidden_states ~ [ bs, seq_len, embedding_dim ]
        # embeddings_attention_mask ~ [ bs, seq_len ]
        audio_embeds, audio_embeds_attention_mask = self.model.encode_audio(batched_waveforms, batched_waveforms_attention_mask)

        assert not audio_embeds.isnan().any()

        assert audio_embeds.shape[0] == audio_embeds_attention_mask.shape[0]
        assert audio_embeds.shape[1] == audio_embeds_attention_mask.shape[1]


        inputs_ids = inputs['input_ids'].to(self.args.device)
        inputs_embeds = self.model.encode_text(inputs_ids)
        attention_mask = inputs['attention_mask'].to(self.args.device)

        model_inputs_with_audio = self.model.prepare_audio_inputs(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            audio_embeds=audio_embeds,
            audio_embeds_attention_mask=audio_embeds_attention_mask,
        )

        return {
            "input_ids": inputs_ids,
            "input_ids_attention_mask": inputs['input_ids_attention_mask'].to(self.args.device),
            "audio_embeds": model_inputs_with_audio['audio_embeds'],
            "audio_embeds_attention_mask": model_inputs_with_audio['audio_embeds_attention_mask'],
            "inputs_embeds":  model_inputs_with_audio["inputs_embeds"],
            "attention_mask": model_inputs_with_audio["attention_mask"],
        }

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """


        outputs: BaseModelOutputWithPast = model(
            inputs_embeds=inputs['inputs_embeds'],
            attention_mask=inputs['attention_mask'],
        )

        batch_input_ids = inputs['input_ids']
        batch_input_ids_attention_mask = inputs['input_ids_attention_mask']
        caption_legth = batch_input_ids.shape[1]
        model_prediction_caption = outputs.logits[:, -caption_legth:-1, :]  # [ bs, caption_length - 1, vocad_size ]

        shifted_batch_input_ids = batch_input_ids[:, 1:]  # [ bs, caption_length - 1 ]
        shifted_input_ids_attention_mask = batch_input_ids_attention_mask[:, 1:]
        assert shifted_batch_input_ids.shape == shifted_input_ids_attention_mask.shape

        model_prediction_caption_flatten = model_prediction_caption.flatten(0, 1)
        input_ids_flatten = shifted_batch_input_ids.flatten(0, 1)
        input_ids_attention_mask_flatten = shifted_input_ids_attention_mask.flatten(0, 1).bool()
        assert model_prediction_caption_flatten.shape[0] == input_ids_flatten.shape[0]

        # do not train to predict pad token
        model_prediction_caption_flatten = model_prediction_caption_flatten[input_ids_attention_mask_flatten]
        input_ids_flatten = input_ids_flatten[input_ids_attention_mask_flatten]

        # TODO label smoothing?
        criterion = CrossEntropyLoss()
        loss = criterion(model_prediction_caption_flatten, input_ids_flatten)

        audio_embeds_len = inputs['audio_embeds_attention_mask'].shape[-1]
        audio_embeddings = inputs['inputs_embeds'][:, :audio_embeds_len, :].flatten(0, 1)[inputs['audio_embeds_attention_mask'].flatten().bool()]
        audio_embeddings_norm_mean = audio_embeddings.norm(2, dim=-1).mean().item()

        audio_embeddings_mean = audio_embeddings.mean(dim=-1).mean().item()

        text_embeddings = inputs['inputs_embeds'][:, audio_embeds_len+1:, :].flatten(0, 1)[inputs['attention_mask'][:, audio_embeds_len+1:].flatten().bool()]
        text_embeddings_norm_mean = text_embeddings.norm(2, dim=-1).mean().item()
        text_embeddings_mean = text_embeddings.mean(dim=-1).mean().item()

        audio_bos_mean = model.audio_tokens_embeddings.weight[0].mean().item()
        audio_bos_norm = model.audio_tokens_embeddings.weight[0].norm(2).item()
        audio_eos_mean = model.audio_tokens_embeddings.weight[1].mean().item()
        audio_eos_norm = model.audio_tokens_embeddings.weight[1].norm(2).item()

        step_metrics = {
            "train_loss": loss.item(),
            "seq_len": inputs['attention_mask'].shape[-1],
            "debug/audio_embeddings_norm_mean": audio_embeddings_norm_mean,
            "debug/text_embeddings_norm_mean": text_embeddings_norm_mean,
            "debug/audio_embeddings_mean": audio_embeddings_mean,
            "debug/text_embeddings_mean": text_embeddings_mean,
            "debug/text_embeddings_mean": text_embeddings_mean,
            "debug/text_embeddings_mean": text_embeddings_mean,
            "debug/audio_bos_mean": audio_bos_mean,
            "debug/audio_bos_norm": audio_bos_norm,
            "debug/audio_eos_mean": audio_eos_mean,
            "debug/audio_eos_norm": audio_eos_norm,
        }

        self.accelerator.log(step_metrics)

        return (loss, outputs) if return_outputs else loss


    # TODO make generation here
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

