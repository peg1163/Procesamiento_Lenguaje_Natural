import argparse
import math
from itertools import cycle

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from .data import create_dataloaders
from .utils import (
    load_config,
    set_seed,
    get_device,
    save_checkpoint,
    count_parameters,
    LOGGER,
)
from .models.transformer import MicroTransformer 
# Scheduler con warmup + decay

def build_scheduler(optimizer, num_steps: int, warmup_steps: int):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        # decay lineal simple
        return max(0.0, float(num_steps - step) / float(max(1, num_steps - warmup_steps)))

    return LambdaLR(optimizer, lr_lambda)



def evaluate(model, data_loader, device, criterion) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)  # (B, T, V)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

    avg_loss = total_loss / max(1, total_tokens)
    return avg_loss



def train(config_path: str, profile: bool = False):
    cfg = load_config(config_path)
    set_seed(cfg["train"].get("seed", 42))

    device = get_device(cfg["train"].get("device", "cuda"))
    LOGGER.info(f"Usando device: {device}")

    # Data
    train_loader, val_loader, tokenizer = create_dataloaders(cfg)
    vocab_size = tokenizer.vocab_size
    LOGGER.info(f"Vocab_size detectado: {vocab_size}")

    # Modelo
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

    LOGGER.info(f"Modelo con {count_parameters(model):,} parámetros.")

    # Optimizador + scheduler
    train_cfg = cfg["train"]
    lr = train_cfg["lr"]
    num_steps = train_cfg["num_steps"]
    warmup_steps = train_cfg.get("warmup_steps", 0)
    grad_clip = train_cfg.get("grad_clip", 1.0)

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )
    scheduler = build_scheduler(optimizer, num_steps=num_steps, warmup_steps=warmup_steps)

    criterion = nn.CrossEntropyLoss() 

    # Loop
    step = 0
    best_val_loss = float("inf")
    train_iter = cycle(train_loader)

    if profile:
        LOGGER.info("Profiling activado: solo unos pocos pasos serán perfilados.")

    while step < num_steps:
        model.train()
        x, y = next(train_iter)
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)

        loss = criterion(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
        )
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        step += 1

        if step % 50 == 0 or step == 1:
            LOGGER.info(f"Step {step}/{num_steps} - loss={loss.item():.4f}")

        # evaluación periódica
        if step % train_cfg.get("eval_every", 500) == 0 or step == num_steps:
            val_loss = evaluate(model, val_loader, device, criterion)
            ppl = math.exp(val_loss)
            LOGGER.info(f"[VAL] step={step} loss={val_loss:.4f} ppl={ppl:.2f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    "checkpoints/best.pt",
                    model,
                    optimizer,
                    scheduler,
                    step=step,
                    extra={"val_loss": val_loss},
                )

        # Profiling (muy básico): profilar 1 paso y salir
        if profile and step == 5:
            LOGGER.info("Iniciando torch.profiler en un solo batch...")
            import torch.profiler as profiler

            with profiler.profile(
                activities=[profiler.ProfilerActivity.CPU] +
                ([profiler.ProfilerActivity.CUDA] if device.type == "cuda" else []),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                # un batch de ejemplo
                x_p, y_p = next(train_iter)
                x_p = x_p.to(device)
                model(x_p)

            prof.export_chrome_trace("trace_train.json")
            LOGGER.info("Perfil guardado en trace_train.json")

    LOGGER.info("Entrenamiento finalizado.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    train(args.config, profile=args.profile)


if __name__ == "__main__":
    main()
