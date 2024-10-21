import torch

from scripts.training import val_loop, TrainConfig, get_model, get_val_dataloader, get_dataloaders

if __name__ == "__main__":

    train_config = TrainConfig()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, tokenizer = get_model(train_config, slm_from_pretrained="data/models/royal-haze-19/last/", device=device)

    train_dataloader, val_dataloader = get_dataloaders(train_config, tokenizer)

    validation_metrics = val_loop(model, tokenizer, val_dataloader, epoch=0, device=device)

    print(validation_metrics)