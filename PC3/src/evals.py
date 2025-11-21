import argparse
import math

import torch
import torch.nn as nn

from .data import create_dataloaders
from .models.transformer import MicroTransformer
from .utils import load_config, set_seed, get_device, load_checkpoint, safe_exp, LOGGER


def evaluate_split(config_path: str, split: str, checkpoint: str):
    cfg = load_config(config_path)
    set_seed(cfg["train"].get("seed", 42))
    device = get_device(cfg["train"].get("device", "cuda"))

    # Data 
    train_loader, val_loader, tokenizer = create_dataloaders(cfg)
    vocab_size = tokenizer.vocab_size

    model_cfg = cfg["model"]
    model = MicroTransformer(
        vocab_size=vocab_size,
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        d_ff=model_cfg["d_ff"],
        dropout=model_cfg.get("dropout", 0.1),
        max_seq_len=model_cfg["max_seq_len"],
        posenc_type=model_cfg.get("posenc_type", "rope"),
    ).to(device)

    load_checkpoint(checkpoint, model, map_location=str(device))

    loader = val_loader if split == "val" else train_loader

    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

    avg_loss = total_loss / max(1, total_tokens)
    ppl = safe_exp(avg_loss)

    LOGGER.info(f"Split={split} loss={avg_loss:.4f} ppl={ppl:.2f}")
    print(f"{split} loss={avg_loss:.4f} ppl={ppl:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    args = parser.parse_args()

    evaluate_split(args.config, args.split, args.checkpoint)


if __name__ == "__main__":
    main()
