import torch

from scripts.trainer_train import build_model, build_dataloaders
from aat.training.validate import val_loop

from aat.training.config import TrainConfig, projection_training

import evaluate

if __name__ == "__main__":

    train_config = projection_training()
    train_config.val_batch_size = 20
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, tokenizer = build_model(train_config, from_pretrained="data/models/checkpoint-16872", device=device)

    train_dataloader, val_dataloader = build_dataloaders(train_config, tokenizer)

    captioning_metrics = evaluate.combine(
        [
            evaluate.load("bleu", keep_in_memory=True),
            evaluate.load("rouge", keep_in_memory=True),
            evaluate.load("meteor", keep_in_memory=True),
        ]
    )

    wer_compute = evaluate.load("wer")

    validation_metrics = val_loop(train_config, model, tokenizer, val_dataloader, epoch=0, device=device, wer_compute=wer_compute, captioning_metrics=captioning_metrics, no_loss=True)

    print(validation_metrics)