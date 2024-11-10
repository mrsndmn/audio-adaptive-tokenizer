from typing import Dict

import time
from time import perf_counter

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import accelerate

from datasets import IterableDataset

from aat.model import AslmModel
from aat.training.config import TrainConfig
from aat.training.batch_prepare import prepare_model_inputs_from_batch

def train_loop(accelerator: accelerate.Accelerator, train_config: TrainConfig, model: AslmModel, optimizer, optimizer_lr_scheduler, train_dataloader: DataLoader, epoch, criterion, last_validation_wer=0.0, device=None):
    model.train()
    progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch}')

    if isinstance(train_dataloader.dataset, IterableDataset):
        # reshuffle steaming dataset
        train_dataloader.dataset.set_epoch(epoch)

    for batch_i, batch in enumerate(train_dataloader):
        with accelerator.accumulate(model):
            model_inputs_with_audio = prepare_model_inputs_from_batch(train_config, model, batch, device=device)

            model_prediction = model.forward(**model_inputs_with_audio)

            # model_prediction
            batch_input_ids = batch['input_ids'].to(device)
            batch_input_ids_attention_mask = batch['input_ids_attention_mask'].to(device)
            caption_legth = batch_input_ids.shape[1]
            model_prediction_caption = model_prediction.logits[:, -caption_legth:-1, :]  # [ bs, caption_length - 1, vocad_size ]

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

            loss = criterion(model_prediction_caption_flatten, input_ids_flatten)

            audio_embeds_len = batch['audio_embeds_attention_mask'].shape[-1]
            audio_embeddings = model_inputs_with_audio['inputs_embeds'][:, :audio_embeds_len, :].flatten(0, 1)[batch['audio_embeds_attention_mask'].flatten().bool()]
            audio_embeddings_norm_mean = audio_embeddings.norm(2, dim=-1).mean().item()

            audio_embeddings_mean = audio_embeddings.mean(dim=-1).mean().item()

            text_embeddings = model_inputs_with_audio['inputs_embeds'][:, audio_embeds_len+1:, :].flatten(0, 1)[model_inputs_with_audio['attention_mask'][:, audio_embeds_len+1:].flatten().bool()]
            text_embeddings_norm_mean = text_embeddings.norm(2, dim=-1).mean().item()
            text_embeddings_mean = text_embeddings.mean(dim=-1).mean().item()

            audio_bos_mean = model.audio_tokens_embeddings.weight[0].mean().item()
            audio_bos_norm = model.audio_tokens_embeddings.weight[0].norm(2).item()
            audio_eos_mean = model.audio_tokens_embeddings.weight[1].mean().item()
            audio_eos_norm = model.audio_tokens_embeddings.weight[1].norm(2).item()

            step_metrics = {
                "train_loss": loss.item(),
                "epoch": epoch,
                "seq_len": model_inputs_with_audio['attention_mask'].shape[-1],
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

            if train_config.debug_attentions:
                model_prediction_attentions = model_prediction.attentions
                for layer_i in range(len(model_prediction_attentions)):
                    layer_attention = model_prediction_attentions[layer_i]
                    audio_weights = layer_attention[:, :, :audio_embeds_len, :audio_embeds_len]
                    text_weights = layer_attention[:, :, audio_embeds_len:, audio_embeds_len:]

                    audio_weights_flatten = audio_weights.flatten()[audio_weights.flatten() != 0]
                    text_weights_flatten = text_weights.flatten()[text_weights.flatten() != 0]
                    step_metrics[f'debug_attention_{layer_i}/audio_mean_weight'] = audio_weights_flatten.mean()
                    step_metrics[f'debug_attention_{layer_i}/text_mean_weight'] = text_weights_flatten.mean()
                    step_metrics[f'debug_attention_{layer_i}/audio_sum_weight'] = audio_weights_flatten.sum()
                    step_metrics[f'debug_attention_{layer_i}/text_sum_weight'] = text_weights_flatten.sum()

            model.zero_grad()

            # loss.backward()
            accelerator.backward(loss)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)

            optimizer.step()
            if optimizer_lr_scheduler is not None:
                optimizer_lr_scheduler.step()
                step_metrics['learning_rate'] = optimizer_lr_scheduler.get_last_lr()[0]

            progress_bar.update(1)
            progress_bar.set_description(f'Epoch={epoch} Loss={loss.item():.3f} WER={last_validation_wer:.3f}')

            accelerator.log(step_metrics)
            # end train loop

    return

