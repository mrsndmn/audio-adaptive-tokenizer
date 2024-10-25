import yaml
import argparse

import pathlib
import random
import torch
import torch.nn as nn

import numpy as np

import logging
import evaluate

import time
from typing import List, Dict

import datasets
from transformers.generation import GenerationConfig

from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
import transformers
from transformers import AutoModel, AutoProcessor, AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaConfig, AutoConfig

from tqdm.auto import tqdm

import wandb
from wandb import sdk as wandb_sdk

import accelerate

from aat.model import TokenizedSpeechLM
from aat.lr_scheduler import WarmupLRScheduler
from torch.optim.lr_scheduler import CyclicLR

from aat.tokenizer import AdaptiveAudioAmplitudeTokenizer
from aat.audio import AudioWaveform

from speechtokenizer import SpeechTokenizer


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')


class TrainConfig:
    log_level = "DEBUG"
    # Training
    num_epochs = 100
    train_batch_size = 20
    val_batch_size = 1
    log_grad_norm = True
    learning_rate = 3e-4
    lm_learning_rate = 1e-4
    # gradient_accumulation_steps = 2

    optimize_audio_encoder = False

    evaluate_every_epoch_mod = 5
    save_model_every_epoch_mod = 5

    no_validation = True

    sampling_rate = 16000

    # Model
    audio_encoder_pretrained_model = "facebook/hubert-large-ls960-ft"
    lm_pretrained_model = "HuggingFaceTB/SmolLM-135M-Instruct"
    lm_simple_model = False # only 2 layers
    optim_lm = False

    # Data
    few_train_samples = None
    few_val_samples = 10
    dataloader_num_workers = 0

    train_dataset_path = "./data/segments_tokenized_1_of_64.dataset/"
    validation_dataset_path = "./data/segments_tokenized_1_of_64.dataset/"

    # train_dataset_path = "./data/segments.dataset"
    # validation_dataset_path = "./data/segments.dataset"


def prepare_model_inputs_from_batch(model: TokenizedSpeechLM, batch, device=None):
    audio_embeds_last_hidden_state = batch['audio_embeds_last_hidden_state'].to(device)
    audio_embeds_attention_mask = batch['audio_embeds_attention_mask'].to(device)

    inputs_embeds = model.encode_text(batch['input_ids'].to(device))
    attention_mask = batch['attention_mask'].to(device)

    model_inputs_with_audio = model.prepare_audio_inputs(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        audio_embeds=audio_embeds_last_hidden_state,
        audio_embeds_attention_mask=audio_embeds_attention_mask,
    )

    return {
        "inputs_embeds":  model_inputs_with_audio["inputs_embeds"],
        "attention_mask": model_inputs_with_audio["attention_mask"],
    }


def save_model(train_config: TrainConfig, model: TokenizedSpeechLM, path: pathlib.Path):
    path.mkdir(parents=True, exist_ok=True)
    logger.info(f"save model to {path}")

    model.save_pretrained(path)

    return

from evaluate import load
wer = load("wer")

@torch.no_grad()
def compute_validation_metrics(generations, references, captioning_metrics=None):

    wer_references = [ x[0] for x in references ]
    print("generations", generations)
    print("wer_references", wer_references)
    wer_score = wer.compute(predictions=generations, references=wer_references)

    validation_metrics = {
        "validation/wer": wer_score
    }

    try:
        if captioning_metrics is not None:
            evaluate_bleu_results = captioning_metrics.compute(predictions=generations, references=references)
            logger.info(f"evaluate_bleu_results {evaluate_bleu_results}")

            validation_metrics["validation/evaluate_bleu"] = evaluate_bleu_results['bleu'] * 100
            validation_metrics["validation/evaluate_rouge1"] = evaluate_bleu_results['rouge1']
            validation_metrics["validation/evaluate_rouge2"] = evaluate_bleu_results['rouge2']
            validation_metrics["validation/evaluate_rougeL"] = evaluate_bleu_results['rougeL']
            validation_metrics["validation/evaluate_rougeLsum"] = evaluate_bleu_results['rougeLsum']
            validation_metrics["validation/evaluate_meteor"] = evaluate_bleu_results['meteor']
    except Exception as e:
        print("Catch eval exception", e)

    return validation_metrics


@torch.no_grad()
def val_loop(train_config: TrainConfig, model: TokenizedSpeechLM, tokenizer, val_dataloader: DataLoader, epoch, no_loss=False, device=None, captioning_metrics=None):

    if train_config.no_validation:
        return {}

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    logger.info(f"go validation {epoch}")
    sumloss = 0
    num_batches = 0

    generations = []
    target_generations = []

    gen_params = {
        "do_sample": False,
        "early_stopping": True,
        "num_beams": 3,
        "repetition_penalty": 2.5,
        "remove_invalid_values": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
        "forced_eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "no_repeat_ngram_size": 4,
        "num_return_sequences": 1,
    }
    genconfig = GenerationConfig.from_model_config(model.lm_decoder.config)

    model.eval()
    for batch in tqdm(val_dataloader):

        batch_input_ids = batch['input_ids'].to(device)
        caption_legth = batch_input_ids.shape[1]

        if not no_loss:
            _inplace_audio_encode_batch_speechtokenizer(train_config, model, batch)

            model_inputs_with_audio = prepare_model_inputs_from_batch(model, batch, device=device)

            model_prediction = model(**model_inputs_with_audio)

            model_prediction_caption = model_prediction.logits[:, -caption_legth:-1, :]  # [ bs, caption_length - 1, vocad_size ]
            shifted_batch_input_ids = batch_input_ids[:, 1:]  # [ bs, caption_length - 1 ]

            model_prediction_caption_flatten = model_prediction_caption.flatten(0, 1)
            input_ids_flatten = shifted_batch_input_ids.flatten(0, 1)
            loss = criterion(model_prediction_caption_flatten, input_ids_flatten)

            sumloss += loss.item()
            num_batches += 1

        audio_embeds_last_hidden_state = batch['audio_embeds_last_hidden_state'].to(device)
        audio_embeds_attention_mask = batch['audio_embeds_attention_mask'].to(device)

        generations_bos = torch.full([ audio_embeds_last_hidden_state.shape[0], 1 ], tokenizer.bos_token_id, device=device)
        attention_mask_bos = torch.ones_like(generations_bos)
        model_inputs_with_only_audio = model.prepare_audio_inputs(
            input_ids=attention_mask_bos,
            attention_mask=attention_mask_bos,
            audio_embeds=audio_embeds_last_hidden_state,
            audio_embeds_attention_mask=audio_embeds_attention_mask,
        )

        genconfig.max_length = caption_legth

        all_generation_params = {
            'generation_config': genconfig,
            'max_new_tokens': caption_legth,
            **model_inputs_with_only_audio,
            **gen_params,
        }

        model_generation = model.lm_decoder.generate(**all_generation_params)
        generated_sentences = tokenizer.batch_decode(model_generation, skip_special_tokens=True)
        for sentence in generated_sentences:
            sentence: str
            sentence = sentence.replace("\n", " ")
            generations.append(sentence)

        all_references = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        for i, reference in enumerate(all_references):
            reference: str
            reference = reference.replace("\n", " ")
            target_generations.append([ reference ])

    assert len(generations) > 0, f"len(generations)={len(generations)}"
    assert len(target_generations) == len(generations), f"len(target_generations) == len(generations): {len(target_generations)} == {len(generations)}"

    validation_metrics = compute_validation_metrics(generations, target_generations, captioning_metrics=captioning_metrics)
    validation_metrics["validation/loss"] = sumloss / (num_batches + 1e-5)

    return validation_metrics

def _inplace_audio_encode_batch_hubert(train_config: TrainConfig, model: TokenizedSpeechLM, batch):
    audio_hidden_states = model.audio_encoder(
        input_values=batch['audio_input_values'].to(device),
        attention_mask=batch['audio_attention_mask'].to(device),
    ).last_hidden_state

    padding_mask_for_hidden_states = model.audio_encoder._get_feature_vector_attention_mask(audio_hidden_states.shape[1], batch['audio_attention_mask'])
    padding_mask_for_hidden_states = padding_mask_for_hidden_states.to(device)

    audio_hidden_states[~padding_mask_for_hidden_states] = 0.0

    batch['audio_embeds_last_hidden_state'] = audio_hidden_states
    batch['audio_embeds_attention_mask'] = padding_mask_for_hidden_states

    # TODO baseline as all hubert embeddings
    # # [ batch_size, seq_len, 1024 ]
    # pooled_output = audio_hidden_states.sum(dim=1) / padding_mask_for_hidden_states.sum(dim=1, keepdim=True)

    # # [ batch_size, seq_len, 1024 ]
    # pooled_output = pooled_output.unsqueeze(1)

def _inplace_audio_encode_batch_speechtokenizer(train_config: TrainConfig, model: TokenizedSpeechLM, batch):
    # [ 1, BS, seq_len ]
    audio_codes = model.audio_encoder.encode(
        batch['audio_input_values'].unsqueeze(1).to(device),
        n_q=1,
    )
    audio_codes = audio_codes.squeeze(0) # [ BS, seq_len ]

    # [ BS, seq_len, embedding_dim ]
    audio_hidden_states = model.speech_tokenizer_embeddings(audio_codes)

    compression_factor = batch['audio_input_values'].shape[-1] / audio_codes.shape[-1]
    compressed_seq_lengths = torch.round(batch['audio_attention_mask'].sum(dim=-1) / compression_factor).to(torch.long)

    assert (compressed_seq_lengths > 0).all()

    codes_attention_mask = torch.zeros_like(audio_codes)
    for i in range(audio_codes.shape[0]):
        seq_len = compressed_seq_lengths[i]
        codes_attention_mask[:seq_len] = 1

    batch['audio_embeds_last_hidden_state'] = audio_hidden_states
    batch['audio_embeds_attention_mask'] = codes_attention_mask

    # TODO baseline as all hubert embeddings
    # # [ batch_size, seq_len, 1024 ]
    # pooled_output = audio_hidden_states.sum(dim=1) / padding_mask_for_hidden_states.sum(dim=1, keepdim=True)

    # # [ batch_size, seq_len, 1024 ]
    # pooled_output = pooled_output.unsqueeze(1)



def train_loop(accelerator: accelerate.Accelerator, model: TokenizedSpeechLM, optimizer, optimizer_lr_scheduler, optimizer_lm, lm_lr_scheduler: LRScheduler, train_dataloader: DataLoader, epoch, criterion, last_validation_wer=0.0, device=None):
    model.train()
    progress_bar = tqdm(range(len(train_dataloader)), desc=f'Epoch {epoch}')

    last_time = time.time()

    for batch_i, batch in enumerate(train_dataloader):
        # with accelerator.accumulate(model):

        _inplace_audio_encode_batch_speechtokenizer(train_config, model, batch)

        prepare_inputs_time = time.time()
        model_inputs_with_audio = prepare_model_inputs_from_batch(model, batch, device=device)
        # print("model_inputs_with_audio['attention_mask']", model_inputs_with_audio['attention_mask'].shape)

        prepare_inputs_time = time.time() - prepare_inputs_time

        forward_time = time.time()
        model_prediction = model.forward(**model_inputs_with_audio)
        forward_time = time.time() - forward_time

        loss_time = time.time()

        # model_prediction
        batch_input_ids = batch['input_ids'].to(device)
        batch_input_ids_attention_mask = batch['input_ids_attention_mask'].to(device)
        caption_legth = batch_input_ids.shape[1]
        # print("caption_legth", caption_legth, "model_prediction.logits.shape", model_prediction.logits.shape)
        model_prediction_caption = model_prediction.logits[:, -caption_legth:-1, :]  # [ bs, caption_length - 1, vocad_size ]

        shifted_batch_input_ids = batch_input_ids[:, 1:]  # [ bs, caption_length - 1 ]
        shifted_input_ids_attention_mask = batch_input_ids_attention_mask[:, 1:]
        # logger.info(f"model_prediction_caption {model_prediction_caption.shape}")
        # logger.info(f"batch_input_ids {shifted_batch_input_ids.shape}")
        model_prediction_caption_flatten = model_prediction_caption.flatten(0, 1)
        input_ids_flatten = shifted_batch_input_ids.flatten(0, 1)
        input_ids_attention_mask_flatten = shifted_input_ids_attention_mask.flatten(0, 1).bool()

        # do not train to predict pad token
        model_prediction_caption_flatten = model_prediction_caption_flatten[input_ids_attention_mask_flatten]
        input_ids_flatten = input_ids_flatten[input_ids_attention_mask_flatten]

        # print("input_ids_attention_mask_flatten", input_ids_attention_mask_flatten.sum(), '/', input_ids_attention_mask_flatten.numel())
        # print("model_prediction_caption_flatten masked", model_prediction_caption_flatten.shape)
        # print("input_ids_flatten masked", input_ids_flatten.shape)

        loss = criterion(model_prediction_caption_flatten, input_ids_flatten)

        loss_time = time.time() - loss_time

        optim_step_time = time.time()

        audio_embeds_len = batch['audio_embeds_attention_mask'].shape[-1]
        audio_embeddings = model_inputs_with_audio['inputs_embeds'][:, :audio_embeds_len, :].flatten(0, 1)[batch['audio_embeds_attention_mask'].flatten().bool()]
        audio_embeddings_norm_mean = audio_embeddings.norm(2, dim=-1).mean().item()
        audio_embeddings_mean = audio_embeddings.mean(dim=-1).mean().item()

        text_embeddings = model_inputs_with_audio['inputs_embeds'][:, audio_embeds_len+1:, :].flatten(0, 1)[model_inputs_with_audio['attention_mask'][:, audio_embeds_len+1:].flatten().bool()]
        text_embeddings_norm_mean = text_embeddings.norm(2, dim=-1).mean().item()
        text_embeddings_mean = text_embeddings.mean(dim=-1).mean().item()
        # print("text_embeddings_norm_mean", text_embeddings_norm_mean, "audio_embeddings_norm_mean", audio_embeddings_norm_mean)

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

        model.zero_grad()
        projection_grad_norm = 0
        audio_tokens_embeddings_grad_norm = 0
        if isinstance(model.projection[0], nn.Linear) and model.projection[0].weight.grad is not None:
            projection_grad_norm = model.projection[0].weight.grad.norm(2)
        if model.audio_tokens_embeddings.weight.grad is not None:
            audio_tokens_embeddings_grad_norm = model.audio_tokens_embeddings.weight.grad.norm(2)
        step_metrics["zg_grad_norm/projection_grad_norm"] = projection_grad_norm
        step_metrics["zg_grad_norm/audio_tokens_embeddings_grad_norm"] = audio_tokens_embeddings_grad_norm

        if model.lm_decoder.lm_head.weight.grad is not None:
            lm_head_grad_norm = model.lm_decoder.lm_head.weight.grad.norm(2)
            step_metrics["zg_grad_norm/lm_head_grad_norm"] = lm_head_grad_norm

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        # accelerator.backward(loss)

        if isinstance(model.projection[0], nn.Linear) and model.projection[0].weight.grad is not None:
            projection_grad_norm = model.projection[0].weight.grad.norm(2)
        if model.audio_tokens_embeddings.weight.grad is not None:
            audio_tokens_embeddings_grad_norm = model.audio_tokens_embeddings.weight.grad.norm(2)
        step_metrics["before_step_grad_norm/projection_grad_norm"] = projection_grad_norm
        step_metrics["before_step_grad_norm/audio_tokens_embeddings_grad_norm"] = audio_tokens_embeddings_grad_norm

        if model.lm_decoder.lm_head.weight.grad is not None:
            lm_head_grad_norm = model.lm_decoder.lm_head.weight.grad.norm(2)
            step_metrics["before_step_grad_norm/lm_head_grad_norm"] = lm_head_grad_norm

        optimizer.step()
        optimizer_lr_scheduler.step()
        step_metrics['projection_lr'] = optimizer_lr_scheduler.get_last_lr()[0]

        # lm optim step
        if optimizer_lm is not None:
            optimizer_lm.step()
            lm_lr_scheduler.step()
            step_metrics['lm_lr'] = lm_lr_scheduler.get_last_lr()[0]


        progress_bar.update(1)
        progress_bar.set_description(f'Epoch={epoch} Loss={loss.item():.3f} WER={last_validation_wer:.3f}')

        optim_step_time = time.time() - optim_step_time

        total_time = time.time() - last_time
        last_time = time.time()

        step_metrics['timing/prepare_inputs']  = prepare_inputs_time
        step_metrics['timing/forward_time']    = forward_time
        step_metrics['timing/loss_time']       = loss_time
        step_metrics['timing/optim_step_time'] = optim_step_time
        step_metrics['timing/total_time'] = total_time
        step_metrics['timing/unknown_time'] = total_time - (prepare_inputs_time + forward_time + loss_time + optim_step_time)

        if train_config.log_grad_norm:
            projection_grad_norm = 0
            if isinstance(model.projection[0], nn.Linear):
                projection_grad_norm = model.projection[0].weight.grad.norm(2)
            audio_tokens_embeddings_grad_norm = model.audio_tokens_embeddings.weight.grad.norm(2)
            step_metrics["grad_norm/projection_grad_norm"] = projection_grad_norm
            step_metrics["grad_norm/audio_tokens_embeddings_grad_norm"] = audio_tokens_embeddings_grad_norm

            if model.lm_decoder.lm_head.weight.grad is not None:
                lm_head_grad_norm = model.lm_decoder.lm_head.weight.grad.norm(2)
                step_metrics["grad_norm/lm_head_grad_norm"] = lm_head_grad_norm

        metric_logger.log(step_metrics)
        # end train loop

    return


def train(
        model: TokenizedSpeechLM,
        tokenizer: transformers.AutoTokenizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        train_config: TrainConfig,
        metric_logger: wandb_sdk.wandb_run.Run,
        device_placement=True,
        device=None,
        captioning_metrics=None,
        ):

    trainable_projection_parameters = list(model.projection.parameters()) + list(model.audio_tokens_embeddings.parameters())
    
    if hasattr(model, 'speech_tokenizer_embeddings'):
        trainable_projection_parameters += list(model.speech_tokenizer_embeddings.parameters())

    if train_config.optimize_audio_encoder:
        trainable_projection_parameters = list(p for p in model.audio_encoder.parameters() if p.requires_grad)

    trainable_lm_parameters = list(model.lm_decoder.parameters())
    optimizer = Adam(trainable_projection_parameters, lr=train_config.learning_rate)
    optimizer_lr_scheduler = CyclicLR(optimizer, base_lr=1e-4, max_lr=3e-4, step_size_up=500)

    if train_config.optim_lm:
        optimizer_lm = Adam(trainable_lm_parameters, lr=train_config.lm_learning_rate)
        optimizer_lm_lr_scheduler = WarmupLRScheduler(optimizer_lm, warmup_steps=1000)
    else:
        optimizer_lm = None # Adam(trainable_lm_parameters, lr=train_config.lm_learning_rate)
        optimizer_lm_lr_scheduler = None # WarmupLRScheduler(optimizer_lm, warmup_steps=1000)

    # Иногда pad_token_id == eos_token_id,
    # но мы хотим, чтобы модель умела предсказывать eos_token_id
    # ignore_index=tokenizer.pad_token_id
    criterion = nn.CrossEntropyLoss()

    accelerator = accelerate.Accelerator(device_placement=device_placement)
    # accelerator.gradient_accumulation_steps = train_config.gradient_accumulation_steps
    # model, optimizer, optimizer_lm, train_dataloader, val_dataloader = accelerator.prepare(model, optimizer, optimizer_lm, train_dataloader, val_dataloader)

    last_validation_wer=0.0

    for epoch in range(train_config.num_epochs):
        train_loop(accelerator, model, optimizer, optimizer_lr_scheduler, optimizer_lm, optimizer_lm_lr_scheduler, train_dataloader, epoch=epoch, criterion=criterion, last_validation_wer=last_validation_wer, device=device)

        if epoch % train_config.evaluate_every_epoch_mod == 0:
            validation_metrics = val_loop(train_config, model, tokenizer, val_dataloader, epoch=epoch, device=device, captioning_metrics=captioning_metrics)
            logger.info(f"validation metrics {validation_metrics}")
            last_validation_wer = validation_metrics.get('validation/wer', 0.0)

            metric_logger.log(validation_metrics)

        if epoch % train_config.save_model_every_epoch_mod == 0:
            base_path_for_model = pathlib.Path(f"data/models/{metric_logger.name}/last/")
            save_model(train_config=train_config, model=model, path=base_path_for_model)

    base_path_for_model = pathlib.Path(f"data/models/{metric_logger.name}/last/")
    save_model(train_config=train_config, model=model, path=base_path_for_model)


def data_preloader():

    def _data_preloader(items):
        result = {
            **items,
            # "audio_embeds_last_hidden_state": [],
        }

        # for audio_embeds_path in items["segments_embeddings_path"]:
        #     audio_embeds_path = audio_embeds_path.replace('/audio_segments_embeddings/', '/audio_segments_embeddings_mean/')
        #     audio_embeds = torch.load(audio_embeds_path, weights_only=True) # [ 1, tokens_count (seq_len), 768 ]
        #     result["audio_embeds_last_hidden_state"].append(audio_embeds)

        return result

    return _data_preloader

N_WORDS = 5

def get_collate_fn(tokenizer, audio_processor, validation=False):

    audio_tokenizer = AdaptiveAudioAmplitudeTokenizer()

    def collate_fn(items):
        result = dict()
        # random select caption
        bos_token = tokenizer.decode(tokenizer.bos_token_id)
        eos_token = tokenizer.decode(tokenizer.eos_token_id)
        tokenizer_input = []
        audio_embeddings_start = []
        audio_embeddings_end = []
        audio_embeddings_lengths = []

        good_items = []
        audio_segments_waveforms = []

        for i, item in enumerate(items):
            words = item['words']
            first_word_second = 0
            last_word_second = item['word_end'][-1]
            if not validation:
                if len(words) > N_WORDS:
                    words_start_idx = random.randint(0, len(words) - N_WORDS)
                    words = words[words_start_idx:words_start_idx+N_WORDS]
                    first_word_second = item['word_start'][words_start_idx]
                    last_word_second = item['word_end'][words_start_idx+N_WORDS-1]
                else:
                    # print("Found low words item:", item['id'], item['words'])
                    continue

            item_text = " ".join(words)
            text_for_item = bos_token + item_text + eos_token

            frame_boarder_left = int(first_word_second * train_config.sampling_rate)
            frame_boarder_right = int(last_word_second * train_config.sampling_rate)

            awf_sr = AudioWaveform(item['audio']['array'], sampling_rate=item['audio']['sampling_rate'])
            item_audio_segments = audio_tokenizer.tokenize(awf_sr)
            segments_frames = [ ias.waveform.shape[-1] for ias in item_audio_segments ]
            # print("segments_frames", segments_frames)

            frames_boarders = np.array(segments_frames).cumsum()

            segment_index_left = np.searchsorted(frames_boarders, frame_boarder_left)

            segment_index_right = np.searchsorted(frames_boarders, frame_boarder_right) + 1
            segment_index_right = min(segment_index_right, len(segments_frames)-1)

            assert segment_index_right > segment_index_left

            item_seq_len = segment_index_right - segment_index_left
            if not validation and item_seq_len > 20:
                print("Too long segment:", item['id'], words_start_idx, words, first_word_second, last_word_second, item_seq_len)
                continue

            tokenizer_input.append(text_for_item)
            audio_embeddings_lengths.append(item_seq_len)
            audio_embeddings_start.append(segment_index_left)
            audio_embeddings_end.append(segment_index_right)

            segment_frame_start = frames_boarders[segment_index_left]
            segment_frame_end = frames_boarders[segment_index_right]
            assert segment_frame_start < segment_frame_end

            audio_segments_waveforms.append(item['audio']['array'][segment_frame_start:segment_frame_end])

            good_items.append(item)


        tokenized_caption = tokenizer(tokenizer_input, padding=True)
        result['input_ids'] = torch.tensor(tokenized_caption['input_ids'])
        result['attention_mask'] = torch.tensor(tokenized_caption['attention_mask'])

        result['input_ids_attention_mask'] = result['attention_mask']

        audio_preprocessed = audio_processor(
            audio_segments_waveforms,
        )

        result['audio_input_values'] = audio_preprocessed['input_values']
        result['audio_attention_mask'] = audio_preprocessed['attention_mask']
        assert result['audio_input_values'].shape[1] > 0
        assert result['audio_attention_mask'].shape[1] > 0

        return result

    return collate_fn


def get_train_dataloader(audio_stt_dataset, train_config: TrainConfig, tokenizer, audio_processor):

    if train_config.few_train_samples is not None:
        audio_stt_dataset = audio_stt_dataset.select(range(train_config.few_train_samples))

    # print("train", list(x['text'] for x in audio_stt_dataset))

    # audio_stt_dataset.set_transform(data_preloader())
    return DataLoader(audio_stt_dataset, collate_fn=get_collate_fn(tokenizer, audio_processor), batch_size=train_config.train_batch_size, num_workers=train_config.dataloader_num_workers, shuffle=True, drop_last=True)


def get_val_dataloader(audio_stt_dataset, train_config: TrainConfig, tokenizer, audio_processor):

    if train_config.few_val_samples is not None:
        audio_stt_dataset = audio_stt_dataset.select(range(train_config.few_val_samples))

    # print("val", list(x['text'] for x in audio_stt_dataset))
    # audio_stt_dataset.set_transform(data_preloader())

    return DataLoader(audio_stt_dataset,
                      collate_fn=get_collate_fn(tokenizer, audio_processor, validation=True),
                      batch_size=train_config.val_batch_size)


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False
    return

def unfreeze_model(model):
    for p in model.parameters():
        p.requires_grad = True
    return

def get_lm_decoder(train_config: TrainConfig, from_pretrained=None, device=None):

    if train_config.lm_simple_model:
        lm_decoder_config = AutoConfig.from_pretrained(train_config.lm_pretrained_model)
        lm_decoder_config.num_hidden_layers = 2
        lm_decoder = LlamaForCausalLM(lm_decoder_config)
    else:
        print("from_pretrained", from_pretrained)
        lm_decoder = LlamaForCausalLM.from_pretrained(from_pretrained)

    lm_decoder.to(device)

    return lm_decoder

def waveform_padding(waveforms_padding_list: List[np.array]) -> Dict:
    assert len(waveforms_padding_list[0].shape) == 1, 'channel dim is not supported for waveform'
    max_len = max(x.shape[-1] for x in waveforms_padding_list)
    batch_size = len(waveforms_padding_list)

    attention_mask = torch.zeros(batch_size, max_len)
    batched_waveform = torch.zeros(batch_size, max_len)

    for i, wf in enumerate(waveforms_padding_list):
        attention_mask[i, :wf.shape[-1]] = 1
        batched_waveform[i, :wf.shape[-1]] = torch.from_numpy(wf)


    return {
        "input_values": batched_waveform,
        "attention_mask": attention_mask,
    }


def get_audio_encoder(train_config: TrainConfig):

    # processor = AutoProcessor.from_pretrained(train_config.audio_encoder_pretrained_model)
    # model = AutoModel.from_pretrained(train_config.audio_encoder_pretrained_model)
    config_path = 'data/speechtokenizer/config.json'
    ckpt_path = 'data/speechtokenizer/ckpt.dev'
    model = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
    model.eval()

    return model, waveform_padding


def get_model(train_config: TrainConfig, from_pretrained=None, device=None):

    # lm_decoder = LlamaForCausalLM.from_pretrained("data/models/hearty-shadow-9/last")

    tokenizer = AutoTokenizer.from_pretrained(train_config.lm_pretrained_model)
    tokenizer.add_bos_token = True
    tokenizer.add_eos_token = True

    audio_encoder, audio_processor = get_audio_encoder(train_config)

    if from_pretrained is not None:
        lm_decoder = get_lm_decoder(train_config, from_pretrained=from_pretrained, device=device)

        model = TokenizedSpeechLM.from_pretrained(audio_encoder, lm_decoder, from_pretrained)
    else:
        lm_decoder = get_lm_decoder(train_config, from_pretrained=train_config.lm_pretrained_model, device=device)
        model = TokenizedSpeechLM(audio_encoder, lm_decoder)

        model.reinitialize_weights()

    # Qwen crutch
    # lm_decoder.config.bos_token_id = 151644
    # lm_decoder.config.eos_token_id = 151645
    # tokenizer.bos_token_id = 151644
    # tokenizer.eos_token_id = 151645

    model.to(device)
    freeze_model(model.lm_decoder)
    freeze_model(model.audio_encoder)
    # unfreeze_model(model.audio_encoder)

    return model, tokenizer, audio_processor


def get_dataloaders(train_config: TrainConfig, tokenizer, audio_processor):
    # full_audio_stt_dataset: datasets.Dataset = datasets.load_from_disk(train_config.train_dataset_path)
    # train_test_audio_stt_dataset = full_audio_stt_dataset.train_test_split(test_size=1000, seed=1)

    dataset_files = [ f'libris/train-{i:05}-of-00064.parquet' for i in range(1) ] # 1 shard = 1 gb of data
    print("dataset_files", dataset_files)
    audio_dataset = datasets.load_dataset("nguyenvulebinh/asr-alignment", split=datasets.Split.TRAIN, data_files=dataset_files, streaming=False)
    # audio_dataset = load_dataset("nguyenvulebinh/asr-alignment", 'libris', split=datasets.Split.TRAIN, streaming=True)
    audio_dataset.cast_column('audio', datasets.Audio(sampling_rate=train_config.sampling_rate))
    train_test_audio_stt_dataset = audio_dataset.train_test_split(test_size=1000, seed=1)

    logger.info("load train dataloader")
    train_dataloader = get_train_dataloader(
        train_test_audio_stt_dataset['train'], train_config, tokenizer, audio_processor
    )
    logger.info("load val dataloader")
    val_dataloader = get_val_dataloader(
        train_test_audio_stt_dataset['test'], train_config, tokenizer, audio_processor
    )

    return train_dataloader, val_dataloader


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config')

    args = parser.parse_args()

    train_config = TrainConfig()

    if args.config is not None:
        with open(args.config) as config_file:
            logger.info("override config params from %s", args.config)
            training_config_overwrite: dict = yaml.safe_load(config_file)
            if training_config_overwrite is not None:
                for k, v in training_config_overwrite.items():
                    logger.info(f"override config param: {k} {v}")
                    train_config.__setattr__(k, v)

    log_level = logging.getLevelName(train_config.log_level)
    logging.basicConfig(level=log_level, format='%(asctime)s %(levelname)s %(message)s')
    logger.info("loglevel %s", train_config.log_level)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info("load language model")

    model, tokenizer, audio_processor = get_model(train_config, device=device)

    logger.info("model was loaded")

    train_dataloader, val_dataloader = get_dataloaders(train_config, tokenizer, audio_processor)

    logger.info("run training")

    captioning_metrics = evaluate.combine(
        [
            evaluate.load("bleu", keep_in_memory=True),
            evaluate.load("rouge", keep_in_memory=True),
            evaluate.load("meteor", keep_in_memory=True),
        ]
    )

    with wandb.init(project="tokenized_speech_lm") as metric_logger:

        trainable_parameters_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_parameters_count = sum(p.numel() for p in model.parameters())
        logger.info(f"trainable model parameters: {trainable_parameters_count}")
        logger.info(f"total model parameters: {total_parameters_count}")

        train(
            model=model,
            tokenizer=tokenizer,
            metric_logger=metric_logger,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            train_config=train_config,
            captioning_metrics=captioning_metrics,
            device=device,
            device_placement=True,
        )