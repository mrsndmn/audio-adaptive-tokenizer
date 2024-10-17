import yaml
import argparse

import pathlib
import random
import torch
import torch.nn as nn

import logging
import evaluate
from aac_metrics import Evaluate

import datasets
from transformers.generation import GenerationConfig

from torch.optim import Adam
from torch.utils.data import DataLoader
import transformers

from tqdm.auto import tqdm

import wandb
from wandb import sdk as wandb_sdk

import accelerate


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')


class TrainConfig:
    log_level = "DEBUG"
    # Training
    num_epochs = 5
    train_batch_size = 1
    val_batch_size = 1
    log_grad_norm = True
    learning_rate = 1e-4
    gradient_accumulation_steps = 1

    evaluate_every_epoch_mod = 4
    save_model_every_epoch_mod = 1

    # Model
    llaaa_from_pretrained = None
    modality_tokens = 32
    llm_train_lora = False
    llm_lora_from_pretrained = None

    # Data
    few_train_samples = None
    few_val_samples = None
    dataloader_num_workers = 0

    train_dataset_path = "data/clotho1/CLOTHO_v2.1/clotho_hf_dataset/clotho_development.dataset/"
    melspec_train_prefix = "data/clotho1/CLOTHO_v2.1/clotho_melspec_processed_imagebind_single/development/"
    audio_embeds_train_prefix = "data/clotho1/CLOTHO_v2.1/clotho_audio_embeds_processed_imagebind_single/development/"

    val_dataset_path = "data/clotho1/CLOTHO_v2.1/clotho_hf_dataset/clotho_validation.dataset/"
    melspec_val_prefix = "data/clotho1/CLOTHO_v2.1/clotho_melspec_processed_imagebind_single/validation/"
    audio_embeds_val_prefix = "data/clotho1/CLOTHO_v2.1/clotho_audio_embeds_processed_imagebind_single/validation/"


def prepare_model_inputs_from_batch(model: Llaaa, batch):
    if 'audio_embeds_last_hidden_state' in batch:
        audio_embeds_last_hidden_state = batch['audio_embeds_last_hidden_state'].to(model.device)
    else:
        audio_melspec_values = batch['pixel_values'].to(model.device)
        audio_embeds_last_hidden_state = model.encode_audio(audio_melspec_values)

    inputs_embeds = model.encode_text(batch['input_ids'].to(model.device))

    model_inputs_with_audio = model.prepare_audio_inputs(
        inputs_embeds=inputs_embeds,
        attention_mask=batch['attention_mask'].to(model.device),
        audio_embeds=audio_embeds_last_hidden_state,
    )

    return {
        "inputs_embeds":  model_inputs_with_audio["inputs_embeds"],
        "attention_mask": model_inputs_with_audio["attention_mask"],
    }


def get_audio_embeds_last_hidden_state(model, batch):
    if 'audio_embeds_last_hidden_state' in batch:
        audio_embeds_last_hidden_state = batch['audio_embeds_last_hidden_state'].to(model.device)
    else:
        audio_melspec_values = batch['pixel_values'].to(model.device)
        audio_embeds_last_hidden_state = model.encode_audio(audio_melspec_values)
    return audio_embeds_last_hidden_state


def save_model(train_config: TrainConfig, model: Llaaa, path: pathlib.Path):
    path.mkdir(parents=True, exist_ok=True)
    logger.info(f"save model to {path}")

    model.save_pretrained(path)
    if train_config.llm_train_lora:
        model.lm_model.save_pretrained(path.joinpath("lora_adapter"))

    return


@torch.no_grad()
def compute_validation_metrics(generations, target_generations, captioning_metrics=None, aac_evaluate=None):
    validation_metrics = {}
    if captioning_metrics is not None:
        evaluate_bleu_results = captioning_metrics.compute(predictions=generations, references=target_generations, smooth=True)
        logger.info(f"evaluate_bleu_results {evaluate_bleu_results}")

        validation_metrics["validation/evaluate_bleu"] = evaluate_bleu_results['bleu'] * 100
        validation_metrics["validation/evaluate_rouge1"] = evaluate_bleu_results['rouge1']
        validation_metrics["validation/evaluate_rouge2"] = evaluate_bleu_results['rouge2']
        validation_metrics["validation/evaluate_rougeL"] = evaluate_bleu_results['rougeL']
        validation_metrics["validation/evaluate_rougeLsum"] = evaluate_bleu_results['rougeLsum']
        validation_metrics["validation/evaluate_meteor"] = evaluate_bleu_results['meteor']

    if aac_evaluate is not None:
        # aac_evaluate_metrics = aac_evaluate(generations, target_generations)
        # logger.info(f"aac_evaluate_metrics {aac_evaluate_metrics}")
        # for metric_name, metric_value in validation_metrics.items():
        #     validation_metrics['validation/' + metric_name] = metric_value
        try:
            aac_evaluate_metrics, _ = aac_evaluate(generations, target_generations)
            logger.info(f"aac_evaluate_metrics {aac_evaluate_metrics}")
            for metric_name, metric_value in aac_evaluate_metrics.items():
                validation_metrics['validation/' + metric_name] = metric_value.item()
        except Exception as e:
            logger.warning(f"can't compute aac evaluate metric: {e}")

    return validation_metrics


@torch.no_grad()
def val_loop(model: Llaaa, tokenizer, val_dataloader: DataLoader, epoch, no_loss=False, captioning_metrics=None, aac_evaluate=None):

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
    genconfig = GenerationConfig.from_model_config(model.lm_model.config)

    model.eval()
    for batch in tqdm(val_dataloader):

        batch_input_ids = batch['input_ids'].to(model.device)
        caption_legth = batch_input_ids.shape[1]

        if not no_loss:
            model_inputs_with_audio = prepare_model_inputs_from_batch(model, batch)

            model_prediction = model(**model_inputs_with_audio)

            model_prediction_caption = model_prediction.logits[:, -caption_legth:-1, :]  # [ bs, caption_length - 1, vocad_size ]
            shifted_batch_input_ids = batch_input_ids[:, 1:]  # [ bs, caption_length - 1 ]

            model_prediction_caption_flatten = model_prediction_caption.flatten(0, 1)
            input_ids_flatten = shifted_batch_input_ids.flatten(0, 1)
            loss = criterion(model_prediction_caption_flatten, input_ids_flatten)

            sumloss += loss.item()
            num_batches += 1

        audio_embeds_last_hidden_state = get_audio_embeds_last_hidden_state(model, batch)

        model_inputs_with_only_audio = model.prepare_audio_inputs(
            audio_embeds=audio_embeds_last_hidden_state,
        )

        genconfig.max_length = caption_legth

        all_generation_params = {
            'generation_config': genconfig,
            'max_new_tokens': caption_legth,
            **model_inputs_with_only_audio,
            **gen_params,
        }

        model_generation = model.lm_model.generate(**all_generation_params)
        generated_sentences = tokenizer.batch_decode(model_generation, skip_special_tokens=True)
        for sentence in generated_sentences:
            sentence: str
            sentence = sentence.replace("\n", " ")
            generations.append(sentence)

        one_audio_references = []
        all_references = tokenizer.batch_decode(batch['all_input_ids'], skip_special_tokens=True)
        assert len(all_references) % 5 == 0, f'len(all_references) {len(all_references)}'
        for i, reference in enumerate(all_references):
            reference: str
            reference = reference.replace("\n", " ")
            one_audio_references.append(reference)
            if (i+1) % 5 == 0:
                target_generations.append(one_audio_references)
                one_audio_references = []

    assert len(generations) > 0, f"len(generations)={len(generations)}"
    assert len(target_generations) == len(generations), f"len(target_generations) == len(generations): {len(target_generations)} == {len(generations)}"

    validation_metrics = compute_validation_metrics(generations, target_generations, captioning_metrics=captioning_metrics, aac_evaluate=aac_evaluate)
    validation_metrics["validation/loss"] = sumloss / (num_batches + 1e-5)

    return validation_metrics


def train_loop(accelerator: accelerate.Accelerator, model: Llaaa, optimizer, train_dataloader: DataLoader, epoch, criterion, last_validation_bleu=0.0):
    model.train()
    progress_bar = tqdm(range(len(train_dataloader)), desc=f'Epoch {epoch}')
    for batch in train_dataloader:
        with accelerator.accumulate(model):
            model_inputs_with_audio = prepare_model_inputs_from_batch(model, batch)
            model_prediction = model.forward(**model_inputs_with_audio)

            # model_prediction
            batch_input_ids = batch['input_ids'].to(model.device)
            caption_legth = batch_input_ids.shape[1]
            model_prediction_caption = model_prediction.logits[:, -caption_legth:-1, :]  # [ bs, caption_length - 1, vocad_size ]

            shifted_batch_input_ids = batch_input_ids[:, 1:]  # [ bs, caption_length - 1 ]
            # logger.info(f"model_prediction_caption {model_prediction_caption.shape}")
            # logger.info(f"batch_input_ids {shifted_batch_input_ids.shape}")
            model_prediction_caption_flatten = model_prediction_caption.flatten(0, 1)
            input_ids_flatten = shifted_batch_input_ids.flatten(0, 1)
            loss = criterion(model_prediction_caption_flatten, input_ids_flatten)

            model.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            progress_bar.update(1)
            progress_bar.set_description(f'Epoch={epoch} Loss={loss.item():.3f} BLEU={last_validation_bleu:.3f}')

            step_metrics = {"train_loss": loss.item(), "epoch": epoch}
            if train_config.log_grad_norm:
                for name, parameter in model.projection.named_parameters():
                    if parameter.grad is not None:
                        parameter_grad_norm = parameter.grad.norm(2).item()
                    else:
                        parameter_grad_norm = 0.0
                    step_metrics[f'grad_norm_{name}'] = parameter_grad_norm

            metric_logger.log(step_metrics)
            # end accelerator accumulation
        # end train loop

    return


def train(
        model: Llaaa,
        tokenizer: transformers.AutoTokenizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        train_config: TrainConfig,
        metric_logger: wandb_sdk.wandb_run.Run,
        device_placement=True,
        ):

    trainable_parameters = list(model.audio_tokens_embeddings.parameters()) + list(model.projection.parameters())
    optimizer = Adam(trainable_parameters, lr=train_config.learning_rate)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    accelerator = accelerate.Accelerator(device_placement=device_placement)
    accelerator.gradient_accumulation_steps = train_config.gradient_accumulation_steps
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader)

    captioning_metrics = evaluate.combine(
        [
            evaluate.load("bleu", keep_in_memory=True),
            evaluate.load("rouge", keep_in_memory=True),
            evaluate.load("meteor", keep_in_memory=True),
        ]
    )
    aac_evaluate = Evaluate(metrics=["bleu_1", "bleu_2", "bleu_3", "bleu_4", "spider", "fense", "vocab", "rouge_l"])

    best_validation_bleu = 0.0
    last_validation_bleu = 0.0
    for epoch in range(train_config.num_epochs):

        train_loop(accelerator, model, optimizer, train_dataloader, epoch=epoch, criterion=criterion, last_validation_bleu=last_validation_bleu)

        if epoch % train_config.evaluate_every_epoch_mod == 0:
            validation_metrics = val_loop(model, tokenizer, val_dataloader, epoch=epoch, captioning_metrics=captioning_metrics, aac_evaluate=aac_evaluate)
            logger.info(f"validation metrics {validation_metrics}")

            last_validation_bleu = validation_metrics['validation/evaluate_bleu']
            metric_logger.log(validation_metrics)

            if last_validation_bleu > best_validation_bleu:
                best_validation_bleu = last_validation_bleu

                base_path_for_best_model = pathlib.Path(f"data/models/{metric_logger.name}/best/")
                save_model(train_config=train_config, model=model, path=base_path_for_best_model)

        if epoch % train_config.save_model_every_epoch_mod == 0:
            base_path_for_model = pathlib.Path(f"data/models/{metric_logger.name}/last/")
            save_model(train_config=train_config, model=model, path=base_path_for_model)

    base_path_for_model = pathlib.Path(f"data/models/{metric_logger.name}/last/")
    save_model(train_config=train_config, model=model, path=base_path_for_model)


def data_preloader(melspec_path_prefix, audio_embeds_path_prefix):

    def _data_preloader(items):
        result = {
            # "pixel_values": [],
            "audio_embeds_last_hidden_state": [],
        }

        for k in items.keys():
            k: str
            if k.startswith('caption_'):
                result[k] = items[k]

        # only for audio encoder finetuning
        # for pixel_value_path in items["melspec_file_name"]:
        #     pixel_value_full_path = pathlib.Path(melspec_path_prefix).joinpath(pixel_value_path)
        #     pixel_value = torch.load(pixel_value_full_path, map_location='cpu')
        #     result["pixel_values"].append(pixel_value)

        for audio_embeds_path in items["audio_embeds_last_hidden_state_file_name"]:
            audio_embeds_full_path = pathlib.Path(audio_embeds_path_prefix).joinpath(audio_embeds_path)
            audio_embeds = torch.load(audio_embeds_full_path, map_location='cpu')
            result["audio_embeds_last_hidden_state"].append(audio_embeds)

        return result

    return _data_preloader


def get_collate_fn(tokenizer, validation=False):
    def collate_fn(items):
        result = dict()
        # random select caption
        current_caption_i = random.randint(1, 5)
        tokenizer_input = [item[f'caption_{current_caption_i}'] for item in items]
        tokenized_caption = tokenizer(tokenizer_input, padding=True)
        result['input_ids'] = torch.tensor(tokenized_caption['input_ids'])
        result['attention_mask'] = torch.tensor(tokenized_caption['attention_mask'])
        # result['pixel_values'] = torch.cat([x['pixel_values'] for x in items], dim=0)
        result['audio_embeds_last_hidden_state'] = torch.cat([x['audio_embeds_last_hidden_state'] for x in items], dim=0)

        if validation:
            all_captions = []
            for item in items:
                for current_caption_i in range(1, 6):
                    all_captions.append(item[f'caption_{current_caption_i}'])

            tokenized_caption = tokenizer(all_captions, padding=True)
            result['all_input_ids'] = torch.tensor(tokenized_caption['input_ids'])
            result['all_attention_mask'] = torch.tensor(tokenized_caption['attention_mask'])
        return result
    return collate_fn


def get_train_dataloader(
        train_config: TrainConfig, llaaa: Llaaa, tokenizer,
        train_dataset_path,
        melspec_train_prefix,
        audio_embeds_train_prefix):

    audio_captions_dataset_train: datasets.Dataset = datasets.load_from_disk(train_dataset_path)
    if train_config.few_train_samples is not None:
        audio_captions_dataset_train = audio_captions_dataset_train.select(range(train_config.few_train_samples))

    audio_captions_dataset_train.set_transform(data_preloader(melspec_train_prefix, audio_embeds_train_prefix))
    return DataLoader(audio_captions_dataset_train, collate_fn=get_collate_fn(tokenizer), batch_size=train_config.train_batch_size, num_workers=train_config.dataloader_num_workers, shuffle=True, drop_last=True)


def get_val_dataloader(
        train_config: TrainConfig, llaaa: Llaaa, tokenizer,
        val_dataset_path,
        melspec_val_prefix,
        audio_embeds_val_prefix):

    audio_captions_dataset_val: datasets.Dataset = datasets.load_from_disk(val_dataset_path)
    if train_config.few_val_samples is not None:
        audio_captions_dataset_val = audio_captions_dataset_val.select(range(train_config.few_val_samples))

    audio_captions_dataset_val.set_transform(data_preloader(melspec_val_prefix, audio_embeds_val_prefix))

    return DataLoader(audio_captions_dataset_val,
                      collate_fn=get_collate_fn(tokenizer, validation=True),
                      batch_size=train_config.val_batch_size)


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False
    return


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

    logger.info("load audio encoder")
    audio_encoder = DummyAudioEncoder()
    audio_encoder.to(device)

    freeze_model(audio_encoder)

    logger.info("load language model")

    if train_config.llm_train_lora:
        lm_model, tokenizer = peft_llama_lm(from_pretrained=train_config.llm_lora_from_pretrained)
    else:
        lm_model, tokenizer = llama_lm()
        freeze_model(lm_model)

    lm_model.to(device)

    logger.info("load Llaaa model")

    if train_config.llaaa_from_pretrained is not None:
        logger.info("load llaaa weights: %s", train_config.llaaa_from_pretrained)
        llaaa_from_pretrained = pathlib.Path(train_config.llaaa_from_pretrained)
        model = Llaaa.from_pretrained(lm_model, audio_encoder, llaaa_from_pretrained)
        # audio_tokens_embeddings = torch.load(llaaa_from_pretrained.joinpath('audio_tokens_embeddings.pt'))
        # projection = torch.load(llaaa_from_pretrained.joinpath('projection.pt'))
        # model.audio_tokens_embeddings.load_state_dict(audio_tokens_embeddings)
        # model.projection.load_state_dict(projection)
    else:
        logger.info("reinitialize weights")

        llaaa_config = LlaaaConfig()
        model = Llaaa(lm_model=lm_model, audio_encoder=audio_encoder, config=llaaa_config)
        model.reinitialize_weights()

    trainable_parameters_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_parameters_count = sum(p.numel() for p in model.parameters())
    logger.info(f"trainable model parameters: {trainable_parameters_count}")
    logger.info(f"total model parameters: {total_parameters_count}")

    logger.info(f"trainable model llm parameters: {sum(p.numel() for p in model.lm_model.parameters() if p.requires_grad)}")
    logger.info(f"trainable model audio encoder parameters: {sum(p.numel() for p in model.audio_encoder.parameters() if p.requires_grad)}")

    logger.info("load train dataloader")
    train_dataloader = get_train_dataloader(
        train_config, model, tokenizer,
        train_dataset_path=train_config.train_dataset_path,
        melspec_train_prefix=train_config.melspec_train_prefix,
        audio_embeds_train_prefix=train_config.audio_embeds_train_prefix,
    )
    logger.info("load val dataloader")
    val_dataloader = get_val_dataloader(
        train_config, model, tokenizer,
        val_dataset_path=train_config.val_dataset_path,
        melspec_val_prefix=train_config.melspec_val_prefix,
        audio_embeds_val_prefix=train_config.audio_embeds_val_prefix,
    )

    logger.info("run training")

    with wandb.init(project="llaaa") as metric_logger:
        train(
            model=model,
            tokenizer=tokenizer,
            metric_logger=metric_logger,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            train_config=train_config,
            # device_placement=False,
        )