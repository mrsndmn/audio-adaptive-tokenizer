import torch

from scripts.training import val_loop, TrainConfig, get_model, get_val_dataloader, get_dataloaders

import evaluate

if __name__ == "__main__":

    train_config = TrainConfig()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, tokenizer = get_model(train_config, from_pretrained="data/models/royal-haze-19/last/", device=device)

    train_dataloader, val_dataloader = get_dataloaders(train_config, tokenizer)

    captioning_metrics = evaluate.combine(
        [
            evaluate.load("bleu", keep_in_memory=True),
            evaluate.load("rouge", keep_in_memory=True),
            evaluate.load("meteor", keep_in_memory=True),
        ]
    )

    validation_metrics = val_loop(model, tokenizer, val_dataloader, epoch=0, device=device, captioning_metrics=captioning_metrics)

    print(validation_metrics)