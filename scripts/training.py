import yaml
import argparse

import pathlib
import random
import torch
import torch.nn as nn

import logging
import evaluate

import datasets
from transformers.generation import GenerationConfig

from torch.optim import Adam
from torch.utils.data import DataLoader
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaConfig, AutoConfig

from tqdm.auto import tqdm

import wandb
from wandb import sdk as wandb_sdk

import accelerate

from aat.model import TokenizedSpeechLM

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')


class TrainConfig:
    log_level = "DEBUG"
    # Training
    num_epochs = 25
    train_batch_size = 20
    val_batch_size = 1
    log_grad_norm = True
    learning_rate = 1e-4
    lm_learning_rate = 1e-4
    gradient_accumulation_steps = 1

    evaluate_every_epoch_mod = 5
    save_model_every_epoch_mod = 5

    # Model
    lm_pretrained_model = "HuggingFaceTB/SmolLM-135M-Instruct"

    # Data
    few_train_samples = None
    few_val_samples = 100
    dataloader_num_workers = 0

    train_dataset_path = "./data/segments_tokenized_10_of_64.dataset"
    validation_dataset_path = "./data/segments_tokenized_10_of_64.dataset"


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

    if captioning_metrics is not None:
        evaluate_bleu_results = captioning_metrics.compute(predictions=generations, references=references)
        logger.info(f"evaluate_bleu_results {evaluate_bleu_results}")

        validation_metrics["validation/evaluate_bleu"] = evaluate_bleu_results['bleu'] * 100
        validation_metrics["validation/evaluate_rouge1"] = evaluate_bleu_results['rouge1']
        validation_metrics["validation/evaluate_rouge2"] = evaluate_bleu_results['rouge2']
        validation_metrics["validation/evaluate_rougeL"] = evaluate_bleu_results['rougeL']
        validation_metrics["validation/evaluate_rougeLsum"] = evaluate_bleu_results['rougeLsum']
        validation_metrics["validation/evaluate_meteor"] = evaluate_bleu_results['meteor']

    return validation_metrics


@torch.no_grad()
def val_loop(model: TokenizedSpeechLM, tokenizer, val_dataloader: DataLoader, epoch, no_loss=False, device=None, captioning_metrics=None):

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

        genconfig.max_length = caption_legth + 10

        all_generation_params = {
            'generation_config': genconfig,
            'max_new_tokens': caption_legth + 10,
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


def train_loop(accelerator: accelerate.Accelerator, model: TokenizedSpeechLM, optimizer, optimizer_lm, train_dataloader: DataLoader, epoch, criterion, last_validation_wer=0.0, device=None):
    model.train()
    progress_bar = tqdm(range(len(train_dataloader)), desc=f'Epoch {epoch}')
    for batch in train_dataloader:
        with accelerator.accumulate(model):
            model_inputs_with_audio = prepare_model_inputs_from_batch(model, batch)
            model_prediction = model.forward(**model_inputs_with_audio)

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

            model.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            optimizer_lm.step()
            progress_bar.update(1)
            progress_bar.set_description(f'Epoch={epoch} Loss={loss.item():.3f} WER={last_validation_wer:.3f}')

            step_metrics = {"train_loss": loss.item(), "epoch": epoch}
            if train_config.log_grad_norm:
                pass

            metric_logger.log(step_metrics)
            # end accelerator accumulation
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
    trainable_lm_parameters = list(model.lm_decoder.parameters())
    optimizer = Adam(trainable_projection_parameters, lr=train_config.learning_rate)
    optimizer_lm = Adam(trainable_lm_parameters, lr=train_config.lm_learning_rate)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    accelerator = accelerate.Accelerator(device_placement=device_placement)
    accelerator.gradient_accumulation_steps = train_config.gradient_accumulation_steps
    model, optimizer, optimizer_lm, train_dataloader, val_dataloader = accelerator.prepare(model, optimizer, optimizer_lm, train_dataloader, val_dataloader)

    last_validation_wer=0.0

    for epoch in range(train_config.num_epochs):

        train_loop(accelerator, model, optimizer, optimizer_lm, train_dataloader, epoch=epoch, criterion=criterion, last_validation_wer=last_validation_wer, device=device)

        if epoch % train_config.evaluate_every_epoch_mod == 0:
            validation_metrics = val_loop(model, tokenizer, val_dataloader, epoch=epoch, device=device, captioning_metrics=captioning_metrics)
            logger.info(f"validation metrics {validation_metrics}")
            last_validation_wer = validation_metrics['validation/wer']

            metric_logger.log(validation_metrics)

        if epoch % train_config.save_model_every_epoch_mod == 0:
            base_path_for_model = pathlib.Path(f"data/models/{metric_logger.name}/last/")
            save_model(train_config=train_config, model=model, path=base_path_for_model)

    base_path_for_model = pathlib.Path(f"data/models/{metric_logger.name}/last/")
    save_model(train_config=train_config, model=model, path=base_path_for_model)


def data_preloader():

    def _data_preloader(items):
        result = {
            "audio_embeds_last_hidden_state": [],
            "text": items['text'],
        }

        for audio_embeds_path in items["segments_embeddings_path"]:
            audio_embeds_path = audio_embeds_path.replace('/audio_segments_embeddings/', '/audio_segments_embeddings_mean/')
            audio_embeds = torch.load(audio_embeds_path, weights_only=True) # [ 1, tokens_count (seq_len), 768 ]
            result["audio_embeds_last_hidden_state"].append(audio_embeds)

        return result

    return _data_preloader


def get_collate_fn(tokenizer, validation=False):
    def collate_fn(items):
        result = dict()
        # random select caption
        bos_token = tokenizer.decode(tokenizer.bos_token_id)
        eos_token = tokenizer.decode(tokenizer.eos_token_id)
        tokenizer_input = [ bos_token + item['text'] + eos_token for item in items]
        tokenized_caption = tokenizer(tokenizer_input, padding=True)
        result['input_ids'] = torch.tensor(tokenized_caption['input_ids'])

        # print("collate result['input_ids']", tokenizer.batch_decode(result['input_ids']))

        result['attention_mask'] = torch.tensor(tokenized_caption['attention_mask'])
        result['input_ids_attention_mask'] = result['attention_mask']

        max_length = max(x['audio_embeds_last_hidden_state'].shape[1] for x in items)
        audio_embeds_hidden_dim = items[0]['audio_embeds_last_hidden_state'].shape[-1]

        audio_embeds_attention_mask = torch.zeros([len(items), max_length])
        collated_audio_embeds_last_hidden_state = torch.zeros([len(items), max_length, audio_embeds_hidden_dim])
        for i, item in enumerate(items):
            seq_len = item['audio_embeds_last_hidden_state'].shape[1]
            audio_embeds_attention_mask[i, :seq_len] = 1
            collated_audio_embeds_last_hidden_state[i:i+1, :seq_len, :] = item['audio_embeds_last_hidden_state']

        result['audio_embeds_attention_mask'] = audio_embeds_attention_mask
        result['audio_embeds_last_hidden_state'] = collated_audio_embeds_last_hidden_state

        return result

    return collate_fn


def get_train_dataloader(audio_stt_dataset, train_config: TrainConfig, tokenizer):

    if train_config.few_train_samples is not None:
        audio_stt_dataset = audio_stt_dataset.select(range(train_config.few_train_samples))

    # print("train", list(x['text'] for x in audio_stt_dataset))

    audio_stt_dataset.set_transform(data_preloader())
    return DataLoader(audio_stt_dataset, collate_fn=get_collate_fn(tokenizer), batch_size=train_config.train_batch_size, num_workers=train_config.dataloader_num_workers, shuffle=True, drop_last=True)


def get_val_dataloader(audio_stt_dataset, train_config: TrainConfig, tokenizer):

    if train_config.few_val_samples is not None:
        audio_stt_dataset = audio_stt_dataset.select(range(train_config.few_val_samples))

    # print("val", list(x['text'] for x in audio_stt_dataset))
    audio_stt_dataset.set_transform(data_preloader())

    return DataLoader(audio_stt_dataset,
                      collate_fn=get_collate_fn(tokenizer, validation=True),
                      batch_size=train_config.val_batch_size)


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False
    return


def get_model(train_config: TrainConfig, from_pretrained=None, device=None):
    # lm_decoder_config = AutoConfig.from_pretrained(train_config.lm_pretrained_model)
    # lm_decoder_config.num_hidden_layers = 2
    # lm_decoder = LlamaForCausalLM(lm_decoder_config)

    # lm_decoder = LlamaForCausalLM.from_pretrained("data/models/hearty-shadow-9/last")

    # freeze_model(lm_decoder)

    tokenizer = AutoTokenizer.from_pretrained(train_config.lm_pretrained_model)
    tokenizer.add_bos_token = True
    tokenizer.add_eos_token = True

    if from_pretrained is not None:
        lm_decoder = LlamaForCausalLM.from_pretrained(from_pretrained)
        lm_decoder.to(device)

        model = TokenizedSpeechLM.from_pretrained(None, lm_decoder, from_pretrained)
    else:
        lm_decoder = LlamaForCausalLM.from_pretrained(train_config.lm_pretrained_model)
        lm_decoder.to(device)
        model = TokenizedSpeechLM(None, lm_decoder)

    model.to(device)

    return model, tokenizer

def get_dataloaders(train_config: TrainConfig, tokenizer):
    full_audio_stt_dataset: datasets.Dataset = datasets.load_from_disk(train_config.train_dataset_path)
    train_test_audio_stt_dataset = full_audio_stt_dataset.train_test_split(test_size=1000, seed=1)

    logger.info("load train dataloader")
    train_dataloader = get_train_dataloader(
        train_test_audio_stt_dataset['train'], train_config, tokenizer,
    )
    logger.info("load val dataloader")
    val_dataloader = get_val_dataloader(
        train_test_audio_stt_dataset['test'], train_config, tokenizer,
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

    model, tokenizer = get_model(train_config, from_pretrained="data/models/gallant-river-24/last", device=device)

    logger.info("model was loaded")

    trainable_parameters_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_parameters_count = sum(p.numel() for p in model.parameters())
    logger.info(f"trainable model parameters: {trainable_parameters_count}")
    logger.info(f"total model parameters: {total_parameters_count}")

    train_dataloader, val_dataloader = get_dataloaders(train_config, tokenizer)

    logger.info("run training")

    captioning_metrics = evaluate.combine(
        [
            evaluate.load("bleu", keep_in_memory=True),
            evaluate.load("rouge", keep_in_memory=True),
            evaluate.load("meteor", keep_in_memory=True),
        ]
    )

    with wandb.init(project="tokenized_speech_lm") as metric_logger:
        train(
            model=model,
            tokenizer=tokenizer,
            metric_logger=metric_logger,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            train_config=train_config,
            device=device,
            device_placement=True,
        )