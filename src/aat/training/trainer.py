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

import numpy as np

from aslm.modeling_aslm import AslmModel
from transformers.modeling_outputs import BaseModelOutputWithPast

from dataclasses import dataclass, field
import transformers
from transformers import GenerationConfig
from transformers.trainer import nested_detach
from transformers.trainer_pt_utils import EvalLoopContainer, find_batch_size, IterableDatasetShard
from transformers.trainer_utils import has_length, denumpify_detensorize, EvalLoopOutput, EvalPrediction

from torch.utils.data import DataLoader

import time

from enum import Enum

class AudioEncoderType(Enum):
    hubert = "hubert"
    efficient_net = "efficient_net"

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = field(default="data/models/hubert_linear_projection_experiments")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    dataloader_drop_last: bool = field(default=True)
    dataloader_num_workers: int = field(default=10)
    per_device_train_batch_size: int = field(default=40)
    gradient_accumulation_steps: int = field(default=2)

    include_for_metrics: List[str] = field(default_factory=lambda: [ 'inputs' ])
    
    num_train_epochs: int = field(default=10)
    few_train_samples: Optional[int] = field(default=None)
    
    eval_steps: int = field(default=1000)
    eval_strategy: str = field(default='steps')
    
    save_total_limit: int = field(default=2)
    save_steps: int = field(default=1000)
    load_best_model_at_end: bool =  field(default=True)

    logging_steps: int = field(default=10)
    
    learning_rate: float = field(default=1e-4)
    
    segmentation: str = field(default="none")
    
    train_audio_encoder: bool =  field(default=True)
    audio_encoder_type: AudioEncoderType =  field(default="hubert")
    audio_encoder_embeddings_seq_len: int = field(default=1)
    max_segment_frames: Optional[int] = field(default=4000)

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
    
    def get_audio_embeds_from_inputs(self, inputs):
        if self.args.train_audio_encoder:
            audio_embeds, audio_embeds_attention_mask = self._get_audio_embeds_from_inputs(inputs)
        else:
            with torch.no_grad():
                audio_embeds, audio_embeds_attention_mask = self._get_audio_embeds_from_inputs(inputs)

        return audio_embeds, audio_embeds_attention_mask
    
    def _get_audio_embeds_from_inputs(self, inputs):
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

        return audio_embeds, audio_embeds_attention_mask

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """

        audio_embeds, audio_embeds_attention_mask = self.get_audio_embeds_from_inputs(inputs)

        inputs_ids = inputs['input_ids'].to(self.args.device)
        inputs_embeds = self.model.encode_text(inputs_ids)
        attention_mask = inputs['attention_mask'].to(self.args.device)

        model_inputs_with_audio = self.model.prepare_audio_inputs(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            audio_embeds=audio_embeds,
            audio_embeds_attention_mask=audio_embeds_attention_mask,
            segments_count=inputs.get('segments_count', None),
        )

        return {
            "input_ids": inputs_ids,
            "input_ids_attention_mask": inputs['input_ids_attention_mask'].to(self.args.device),
            "audio_embeds": model_inputs_with_audio['audio_embeds'],
            "audio_embeds_attention_mask": model_inputs_with_audio['audio_embeds_attention_mask'],
            "inputs_embeds":  model_inputs_with_audio["inputs_embeds"],
            "attention_mask": model_inputs_with_audio["attention_mask"],
            "prefix_input_ids": inputs['prefix_input_ids'].to(self.args.device),
        }

    def compute_loss(self, model, inputs, return_outputs=False, log_metrics=True):
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

        if log_metrics:
            step_metrics = {
                "debug/seq_len": inputs['attention_mask'].shape[-1],
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

    def training_step(self, model: AslmModel, *args, **kwargs):
        result = super().training_step(model, *args, **kwargs)
        
        audio_encdoer_grad = None
        if hasattr(model.audio_encoder, 'feature_projection'):
            audio_encdoer_grad = model.audio_encoder.feature_projection.projection.weight.grad
        elif hasattr(model.audio_encoder, 'efficient_net'):
            audio_encdoer_grad = model.audio_encoder.efficient_net._conv_head.weight.grad

        if self.args.train_audio_encoder:
            assert audio_encdoer_grad is not None, "audio_encdoer_grad is expected to be not none"
            
        audio_tokens_emb_grad = model.audio_tokens_embeddings.weight.grad
        
        extra_log = dict()
        if audio_encdoer_grad is not None:
            extra_log["train/audio_encdoer_grad_norm"] = audio_encdoer_grad.norm(2)

        if audio_tokens_emb_grad is not None:
            extra_log["train/audio_tokens_emb_grad"] = audio_tokens_emb_grad.norm(2)
            
        self.accelerator.log(extra_log)

        return result

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"\n***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_inputs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=self.processing_class.pad_token_id)
        
        metrics = None
        eval_set_kwargs = {}

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = (
                self._prepare_input(inputs[main_input_name]) if "inputs" in args.include_for_metrics else None
            )

            # Update containers
            if losses is not None:
                losses = self.gather_function((losses.repeat(batch_size)))
                all_losses.add(losses)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function((inputs_decode))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_inputs.add(inputs_decode)
            if labels is not None:
                # Pad labels here, preparing for preprocess_logits_for_metrics in next logits block.
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function((logits))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(logits)
            if labels is not None:
                labels = self.gather_function((labels))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(labels)
                    
            extra_eval_set_kwargs = self.update_eval_set_kwargs_containers(model, inputs)
            for key, value in extra_eval_set_kwargs.items():
                if key not in eval_set_kwargs:
                    eval_set_kwargs[key] = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=self.processing_class.pad_token_id)

                eval_set_kwargs[key].add(value)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            if self.args.batch_eval_metrics:
                if self.compute_metrics is not None and logits is not None and labels is not None:
                    is_last_step = self.accelerator.gradient_state.end_of_dataloader
                    batch_kwargs = {}
                    batch_kwargs["losses"] = losses if "loss" in args.include_for_metrics else None
                    batch_kwargs["inputs"] = inputs if "inputs" in args.include_for_metrics else None
                    metrics = self.compute_metrics(
                        predictions=all_preds,
                        label_ids=all_labels,
                        **eval_set_kwargs
                    )

                del losses, logits, labels, inputs, eval_set_kwargs
                torch.cuda.empty_cache()

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            elif args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()
                
                for key in eval_set_kwargs.keys():
                    eval_set_kwargs[key].to_cpu_and_numpy()

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        all_losses = all_losses.get_arrays()
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()
        all_inputs = all_inputs.get_arrays()
        eval_set_kwargs_arrays = dict()
        for key, value in eval_set_kwargs.items():
            eval_set_kwargs_arrays[key] = eval_set_kwargs[key].get_arrays()

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if (
            self.compute_metrics is not None
            # and all_preds is not None
            # and all_labels is not None
            and not self.args.batch_eval_metrics
        ):
            eval_set_kwargs_arrays["losses"] = all_losses if "loss" in args.include_for_metrics else None
            eval_set_kwargs_arrays["inputs"] = all_inputs if "inputs" in args.include_for_metrics else None
            metrics = self.compute_metrics(
                predictions=all_preds,
                label_ids=all_labels,
                **eval_set_kwargs_arrays
            )
        elif metrics is None:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if isinstance(all_losses, list) and all_losses:
            metrics[f"{metric_key_prefix}_loss"] = np.concatenate(all_losses).mean().item()
        elif isinstance(all_losses, np.ndarray):
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time
        if hasattr(self, "model_preparation_time"):
            metrics[f"{metric_key_prefix}_model_preparation_time"] = self.model_preparation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

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
        
        # print("inputs", inputs.keys())
        # breakpoint()
        
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []


        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True, log_metrics=False)
            loss = loss.mean().detach()

            logits = outputs.logits

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        labels = None

        return (loss, logits, labels)

    def update_eval_set_kwargs_containers(self, model, inputs):
        
        audio_embeds_last_hidden_state, audio_embeds_attention_mask = self._get_audio_embeds_from_inputs(inputs)

        # generations_bos = torch.full([ audio_embeds_last_hidden_state.shape[0], 1 ], tokenizer.bos_token_id, device=device)
        # attention_mask_bos = torch.ones_like(generations_bos)
        model_inputs_with_only_audio = model.prepare_audio_inputs(
            input_ids=inputs['prefix_input_ids'],
            attention_mask=inputs['prefix_attention_mask'],
            audio_embeds=audio_embeds_last_hidden_state,
            audio_embeds_attention_mask=audio_embeds_attention_mask,
            segments_count=inputs.get('segments_count', None),
        )

        gen_params = {
            "do_sample": False,
            "early_stopping": True,
            "num_beams": 3,
            "repetition_penalty": 2.5,
            "remove_invalid_values": True,
            "eos_token_id": self.processing_class.eos_token_id,
            "pad_token_id": self.processing_class.eos_token_id,
            "forced_eos_token_id": self.processing_class.eos_token_id,
            "use_cache": True,
            "no_repeat_ngram_size": 4,
            "num_return_sequences": 1,
        }
        genconfig = GenerationConfig.from_model_config(model.lm_decoder.config)

        caption_legth = inputs['input_ids'].shape[1]
        genconfig.max_length = caption_legth

        all_generation_params = {
            'generation_config': genconfig,
            'max_new_tokens': caption_legth,
            'inputs_embeds': model_inputs_with_only_audio['inputs_embeds'],
            'attention_mask': model_inputs_with_only_audio['attention_mask'],
            **gen_params,
        }

        if model.lm_decoder.dtype != all_generation_params['inputs_embeds'].dtype:
            all_generation_params['inputs_embeds'] = all_generation_params['inputs_embeds'].to(model.lm_decoder.dtype)

        model_generation = model.lm_decoder.generate(**all_generation_params)
        
        return {
            "generated_ids": model_generation,
            "prefix_ids": inputs['prefix_input_ids'],
        }


class AATTrainerSegmentation(AATTrainer):

    def _get_audio_embeds_from_inputs(self, inputs):
        # [ bs, segments_count ]
        segments_boarders_padded = inputs['segments_boarders_padded']

        batch_size = segments_boarders_padded.shape[0]
        segments_count = segments_boarders_padded.shape[1]
        
        if self.args.audio_encoder_type == AudioEncoderType.hubert.value:
            # [ bs * segments_count, max_segment_waveform_frames ]
            batched_segments = inputs['batched_segments'].flatten(0,1)
            segments_waveforms_mask = inputs['segments_waveforms_mask'].flatten(0, 1)
        elif self.args.audio_encoder_type == AudioEncoderType.efficient_net.value:
            # [ bs * segments_count, 1, num_mel_features, seq_len ]
            batched_segments = inputs['batched_segments_melspectrograms'].flatten(0,1).unsqueeze(1)
            segments_waveforms_mask = inputs['segments_waveforms_mask'].flatten(0, 1)
        else:
            raise ValueError(f"unknown audio encoder type: {self.args.audio_encoder_type}")

        audio_embeds, audio_embeds_attention_mask = self.model.encode_audio(batched_segments, segments_waveforms_mask)

        # audio_hidden_states ~ [ bs * segments_count, seq_len, embedding_dim ]
        # embeddings_attention_mask ~ [ bs * segments_count, seq_len ]
        
        assert audio_embeds.shape[1] == self.model.config.audio_encoder_embeddings_seq_len

        assert not audio_embeds.isnan().any()

        assert audio_embeds.shape[0] == audio_embeds_attention_mask.shape[0]
        assert audio_embeds.shape[1] == audio_embeds_attention_mask.shape[1]

        return audio_embeds, audio_embeds_attention_mask

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """

        audio_embeds, audio_embeds_attention_mask = self.get_audio_embeds_from_inputs(inputs)
    
        inputs_ids = inputs['input_ids'].to(self.args.device)
        inputs_embeds = self.model.encode_text(inputs_ids)
        attention_mask = inputs['attention_mask'].to(self.args.device)

        model_inputs_with_audio = self.model.prepare_audio_inputs(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            audio_embeds=audio_embeds,
            audio_embeds_attention_mask=audio_embeds_attention_mask,
            segments_count=inputs.get('segments_count', None),
            # TODO is it OK
            segments_boarders_attention_mask=inputs['segments_boarders_attention_mask'],
        )

        return {
            "input_ids": inputs_ids,
            "input_ids_attention_mask": inputs['input_ids_attention_mask'].to(self.args.device),
            "audio_embeds": model_inputs_with_audio['audio_embeds'],
            "audio_embeds_attention_mask": model_inputs_with_audio['audio_embeds_attention_mask'],
            "inputs_embeds":  model_inputs_with_audio["inputs_embeds"],
            "attention_mask": model_inputs_with_audio["attention_mask"],
            "prefix_input_ids": inputs['prefix_input_ids'].to(self.args.device),
        }
