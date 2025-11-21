import argparse
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from .utils import load_config, save_json, load_json, LOGGER


VOCAB_PATH = Path("data") / "vocab.json"


class CharTokenizer:
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos
        self.vocab_size = len(stoi)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        LOGGER.info(f"Vocabulario de {len(stoi)} caracteres construido.")
        return cls(stoi, itos)

    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor([self.stoi[ch] for ch in s], dtype=torch.long)

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path: Path = VOCAB_PATH) -> None:
        data = {"stoi": self.stoi}
        save_json(data, str(path))

    @classmethod
    def load(cls, path: Path = VOCAB_PATH) -> "CharTokenizer":
        data = load_json(str(path))
        stoi = {k: int(v) for k, v in data["stoi"].items()}
        itos = {i: ch for ch, i in stoi.items()}
        LOGGER.info(f"Vocabulario cargado con {len(stoi)} caracteres.")
        return cls(stoi, itos)



class CharDataset(Dataset):
    def __init__(self, ids: torch.Tensor, seq_len: int):
        assert ids.ndim == 1, "ids debe ser un vector 1D"
        self.ids = ids
        self.seq_len = seq_len

    def __len__(self) -> int:
        # Necesitamos al menos seq_len+1 para tener x e y
        return max(0, len(self.ids) - self.seq_len - 1)

    def __getitem__(self, idx: int):
        x = self.ids[idx: idx + self.seq_len]
        y = self.ids[idx + 1: idx + 1 + self.seq_len]
        return x, y




def load_text(path: str) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el corpus: {path}")
    text = path.read_text(encoding="utf-8")
    LOGGER.info(f"Texto cargado desde {path} (longitud {len(text)} caracteres).")
    return text


def build_or_load_tokenizer(text_path: str) -> CharTokenizer:
    if VOCAB_PATH.exists():
        return CharTokenizer.load(VOCAB_PATH)
    text = load_text(text_path)
    tokenizer = CharTokenizer.from_text(text)
    tokenizer.save(VOCAB_PATH)
    return tokenizer


def make_splits(
    ids: torch.Tensor,
    val_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = len(ids)
    n_val = int(n * val_ratio)
    n_train = n - n_val
    train_ids = ids[:n_train]
    val_ids = ids[n_train:]
    LOGGER.info(f"Split datos: train={len(train_ids)}, val={len(val_ids)}")
    return train_ids, val_ids


def create_dataloaders(
    config: dict,
    tokenizer: Optional[CharTokenizer] = None,
) -> Tuple[DataLoader, DataLoader, CharTokenizer]:
    data_cfg = config["data"]
    path = data_cfg["path"]
    seq_len = data_cfg["seq_len"]
    val_ratio = data_cfg.get("val_ratio", 0.01)
    batch_size = config["train"]["batch_size"]

    text = load_text(path)
    tokenizer = tokenizer or build_or_load_tokenizer(path)
    ids = tokenizer.encode(text)

    train_ids, val_ids = make_splits(ids, val_ratio)

    train_ds = CharDataset(train_ids, seq_len)
    val_ds = CharDataset(val_ids, seq_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )
    return train_loader, val_loader, tokenizer



def prepare_from_config(config_path: str) -> None:
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    tokenizer = build_or_load_tokenizer(data_cfg["path"])
    
    LOGGER.info(f"Vocab_size={tokenizer.vocab_size}. "
                f"Recuerda ajustar model.vocab_size o usar tokenizer.vocab_size.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--prepare", action="store_true",
                        help="Solo construye tokenizer y revisa datos.")
    args = parser.parse_args()

    if args.prepare:
        prepare_from_config(args.config)
    else:
        print("Usa --prepare para preparar los datos.")


if __name__ == "__main__":
    main()
