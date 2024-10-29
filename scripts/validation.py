import torch

from scripts.training import val_loop, TrainConfig, build_model, build_val_dataloader, build_dataloaders

import evaluate

if __name__ == "__main__":

    train_config = TrainConfig()
    train_config.val_batch_size = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, tokenizer, audio_processor = build_model(train_config, from_pretrained="data/models/bumbling-vortex-3/last/", device=device)

    train_dataloader, val_dataloader = build_dataloaders(train_config, tokenizer, audio_processor)

    captioning_metrics = evaluate.combine(
        [
            evaluate.load("bleu", keep_in_memory=True),
            evaluate.load("rouge", keep_in_memory=True),
            evaluate.load("meteor", keep_in_memory=True),
        ]
    )

    validation_metrics = val_loop(train_config, model, tokenizer, val_dataloader, epoch=0, device=device, captioning_metrics=captioning_metrics)

    print(validation_metrics)