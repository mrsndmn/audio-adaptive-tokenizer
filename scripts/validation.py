import torch

from scripts.trainer_train import build_model, build_dataloaders, TrainingArguments, AATTrainer
from aat.training.validate import val_loop

from aat.training.config import TrainConfig, projection_training

import evaluate

import argparse
import transformers
import datasets

from accelerate.tracking import filter_trackers
from aat.training.collate import NoSegmentationAudioWaveformCollator
from aat.training.compute_metrics import ComputeMetrics


if __name__ == "__main__":

    train_config = projection_training()
    train_config.val_batch_size = 20
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, tokenizer = build_model(train_config, from_pretrained="data/models/checkpoint-16872", device=device)

    audio_dataset = datasets.load_dataset("nguyenvulebinh/asr-alignment", 'libris')
    audio_dataset_val = audio_dataset['valid'].select(range(100))
    audio_dataset =  audio_dataset['train']
    audio_dataset = audio_dataset.shuffle(seed=42)

    hf_parser = transformers.HfArgumentParser(TrainingArguments)
    (training_args,) = hf_parser.parse_args_into_dataclasses()

    torch.backends.cuda.matmul.allow_tf32 = True

    # captioning_metrics = evaluate.combine(
    #     [
    #         evaluate.load("bleu", keep_in_memory=True),
    #         evaluate.load("rouge", keep_in_memory=True),
    #         evaluate.load("meteor", keep_in_memory=True),
    #     ]
    # )
    # wer_compute = evaluate.load("wer")
    # validation_metrics = val_loop(train_config, model, tokenizer, val_dataloader, epoch=0, device=device, wer_compute=wer_compute, captioning_metrics=captioning_metrics, no_loss=True)
    
    trainer = AATTrainer(
        model,
        training_args,
        processing_class=tokenizer,
        data_collator=NoSegmentationAudioWaveformCollator(train_config, tokenizer),
        train_dataset=audio_dataset,
        eval_dataset=audio_dataset_val,
        # TODO
        compute_metrics=ComputeMetrics(tokenizer),
    )

    metrics = trainer.evaluate()
    print("metrics", metrics)

