import argparse

import pathlib
import torch
import torch.nn as nn

import logging
import evaluate

import datasets

from torch.optim import Adam
from torch.utils.data import DataLoader
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM

import accelerate

from aat.model import TokenizedSpeechLM
from torch.optim.lr_scheduler import CyclicLR

from speechtokenizer import SpeechTokenizer

from aat.tokenizer import AdaptiveAudioAmplitudeTokenizer

from aat.training.config import TrainConfig, overfit_one_batch_train_config, full_unfreeze_train_config
from aat.training.validate import val_loop
from aat.training.train import train_loop

from aat.training.collate import TokenizedAudioWaveformCollator


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def save_model(train_config: TrainConfig, model: TokenizedSpeechLM, path: pathlib.Path):
    path.mkdir(parents=True, exist_ok=True)
    logger.info(f"save model to {path}")

    model.save_pretrained(path)

    return


def train(
        model: TokenizedSpeechLM,
        tokenizer: transformers.AutoTokenizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        train_config: TrainConfig,
        device_placement=True,
        device=None,
        captioning_metrics=None,
        wer_compute=None,
        ):

    optimizer = Adam(model.parameters(), lr=train_config.learning_rate)
    optimizer_lr_scheduler = CyclicLR(optimizer, base_lr=train_config.learning_rate, max_lr=train_config.max_lr, step_size_up=train_config.step_size_up)

    # Иногда pad_token_id == eos_token_id,
    # но мы хотим, чтобы модель умела предсказывать eos_token_id
    # ignore_index=tokenizer.pad_token_id
    criterion = nn.CrossEntropyLoss()

    accelerator = accelerate.Accelerator(device_placement=device_placement, log_with='wandb')

    accelerator.init_trackers(
        project_name="tokenized_speech_lm",
        config=train_config.dict()
    )
    wandb_run = accelerator.get_tracker('wandb')

    # accelerator.gradient_accumulation_steps = train_config.gradient_accumulation_steps
    # model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader)

    last_validation_wer=0.0

    for epoch in range(train_config.num_epochs):
        train_loop(accelerator, train_config, model, optimizer, optimizer_lr_scheduler, train_dataloader, epoch=epoch, criterion=criterion, last_validation_wer=last_validation_wer, device=device)

        if epoch % train_config.evaluate_every_epoch_mod == 0:
            validation_metrics = val_loop(train_config, model, tokenizer, val_dataloader, epoch=epoch, device=device, wer_compute=wer_compute, captioning_metrics=captioning_metrics)
            logger.info(f"validation metrics {validation_metrics}")
            last_validation_wer = validation_metrics.get('validation/wer', 0.0)

            accelerator.log(validation_metrics)

        if epoch % train_config.save_model_every_epoch_mod == 0:
            base_path_for_model = pathlib.Path(f"data/models/{wandb_run.name}/last/")
            save_model(train_config=train_config, model=model, path=base_path_for_model)

    base_path_for_model = pathlib.Path(f"data/models/{wandb_run.name}/last/")
    save_model(train_config=train_config, model=model, path=base_path_for_model)

    accelerator.end_training()



def get_collate_fn(train_config: TrainConfig, validation=False):
    max_segment_duration_milliseconds = int(train_config.max_segment_waveform_frames * 1000 / train_config.sampling_rate)
    audio_tokenizer = AdaptiveAudioAmplitudeTokenizer(
        max_segment_duration_milliseconds=max_segment_duration_milliseconds,
    )

    def build_text_tokenizer():
        return get_tokenizer(train_config)

    return TokenizedAudioWaveformCollator(
        audio_tokenizer,
        build_text_tokenizer,
        sampling_rate=train_config.sampling_rate,
        validation=validation
    )

def get_train_dataloader(audio_stt_dataset, train_config: TrainConfig, tokenizer):

    if train_config.few_train_samples is not None:
        audio_stt_dataset = audio_stt_dataset.select(range(train_config.few_train_samples))

    return DataLoader(audio_stt_dataset, collate_fn=get_collate_fn(train_config),
                      batch_size=train_config.train_batch_size, num_workers=train_config.dataloader_num_workers,
                      drop_last=True, pin_memory=True)


def get_val_dataloader(audio_stt_dataset, train_config: TrainConfig, tokenizer):

    if train_config.few_val_samples is not None:
        audio_stt_dataset = audio_stt_dataset.select(range(train_config.few_val_samples))

    return DataLoader(audio_stt_dataset,
                      collate_fn=get_collate_fn(train_config, validation=True),
                      batch_size=train_config.val_batch_size, pin_memory=True)


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False
    return

def unfreeze_model(model):
    for p in model.parameters():
        p.requires_grad = True
    return

def get_lm_decoder(train_config: TrainConfig, from_pretrained=None, device=None):

    print("from_pretrained", from_pretrained)
    lm_decoder = LlamaForCausalLM.from_pretrained(from_pretrained)

    lm_decoder.to(device)

    return lm_decoder


def get_audio_encoder(train_config: TrainConfig):

    config_path = 'data/speechtokenizer/config.json'
    ckpt_path = 'data/speechtokenizer/ckpt.dev'
    model = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
    model.eval()

    return model


def get_model(train_config: TrainConfig, from_pretrained=None, device=None):

    # lm_decoder = LlamaForCausalLM.from_pretrained("data/models/hearty-shadow-9/last")

    tokenizer = AutoTokenizer.from_pretrained(train_config.lm_pretrained_model)
    tokenizer.add_bos_token = True
    tokenizer.add_eos_token = True

    audio_encoder = get_audio_encoder(train_config)

    if from_pretrained is not None:
        lm_decoder = get_lm_decoder(train_config, from_pretrained=from_pretrained, device=device)

        model = TokenizedSpeechLM.from_pretrained(audio_encoder, lm_decoder, from_pretrained)
    else:
        lm_decoder = get_lm_decoder(train_config, from_pretrained=train_config.lm_pretrained_model, device=device)
        model = TokenizedSpeechLM(audio_encoder, lm_decoder)

        model.reinitialize_weights()

    model.to(device)

    if not train_config.optim_audio_encoder:
        freeze_model(model.audio_encoder)

    if not train_config.optim_lm:
        freeze_model(model.lm_decoder)

    # unfreeze_model(model.audio_encoder)

    return model, tokenizer

def get_tokenizer(train_config: TrainConfig, tokenizer_config=None):
    tokenizer = AutoTokenizer.from_pretrained(train_config.lm_pretrained_model, config=tokenizer_config)
    tokenizer.add_bos_token = True
    tokenizer.add_eos_token = True

    return tokenizer

def get_dataloaders(train_config: TrainConfig, tokenizer):
    # full_audio_stt_dataset: datasets.Dataset = datasets.load_from_disk(train_config.train_dataset_path)
    # train_test_audio_stt_dataset = full_audio_stt_dataset.train_test_split(test_size=1000, seed=1)

    dataset_files = [ f'libris/train-{i:05}-of-00064.parquet' for i in range(train_config.dataset_shards) ] # 1 shard = 1 gb of data
    print("dataset_files", dataset_files)
    if train_config.few_train_samples:
        assert train_config.dataset_shards == 1, 'only one dataset shard is allowed with few_train_samples due to streaming is not possible with few samples'
        audio_dataset = datasets.load_dataset("nguyenvulebinh/asr-alignment", split=datasets.Split.TRAIN, data_files=dataset_files, streaming=False)
    else:
        audio_dataset = datasets.load_dataset("nguyenvulebinh/asr-alignment", split=datasets.Split.TRAIN, data_files=dataset_files, streaming=True)

    test_dataset_files = [ f'libris/train-00063-of-00064.parquet' ] # 1 shard = 1 gb of data
    audio_dataset_test = datasets.load_dataset("nguyenvulebinh/asr-alignment", split=datasets.Split.TRAIN, data_files=test_dataset_files, streaming=False)
    # audio_dataset = load_dataset("nguyenvulebinh/asr-alignment", 'libris', split=datasets.Split.TRAIN, streaming=True)
    audio_dataset.cast_column('audio', datasets.Audio(sampling_rate=train_config.sampling_rate))
    audio_dataset_test.cast_column('audio', datasets.Audio(sampling_rate=train_config.sampling_rate))

    train_test_audio_stt_dataset = audio_dataset_test.train_test_split(test_size=1000, seed=1)

    logger.info("load train dataloader")
    train_dataloader = get_train_dataloader(
        audio_dataset, train_config, tokenizer
    )
    logger.info("load val dataloader")
    val_dataloader = get_val_dataloader(
        train_test_audio_stt_dataset['test'], train_config, tokenizer
    )

    return train_dataloader, val_dataloader


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config')

    args = parser.parse_args()

        #  2. Capture a dictionary of hyperparameters
    # with open(args.config, 'r') as f:
    #     config_json_data = f.read()
    # train_config = TrainConfig.model_validate_json(config_json_data)

    train_config = full_unfreeze_train_config()

    device = train_config.nn_device
    logger.info(f"device {device}")

    logger.info("load language model")

    model, tokenizer = get_model(train_config, device=device)

    logger.info("model was loaded")

    train_dataloader, val_dataloader = get_dataloaders(train_config, tokenizer)

    logger.info("run training")

    captioning_metrics = evaluate.combine(
        [
            evaluate.load("bleu", keep_in_memory=True),
            evaluate.load("rouge", keep_in_memory=True),
            evaluate.load("meteor", keep_in_memory=True),
        ]
    )
    wer_compute = evaluate.load("wer")

    # Initialise your wandb run, passing wandb parameters and any config information

    trainable_parameters_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_parameters_count = sum(p.numel() for p in model.parameters())
    logger.info(f"trainable model parameters: {trainable_parameters_count}")
    logger.info(f"total model parameters: {total_parameters_count}")

    train(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        train_config=train_config,
        captioning_metrics=captioning_metrics,
        wer_compute=wer_compute,
        device=device,
        device_placement=True,
    )

