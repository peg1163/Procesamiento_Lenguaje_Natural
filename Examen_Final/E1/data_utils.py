import random
from typing import Dict, List

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase


def load_lines(path: str) -> List[str]:

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise ValueError(f"No se encontraron líneas de texto en {path}")
    return lines


def split_texts(
    texts: List[str],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:

    if not 0 < train_frac < 1:
        raise ValueError("train_frac debe estar entre 0 y 1.")
    if not 0 < val_frac < 1:
        raise ValueError("val_frac debe estar entre 0 y 1.")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac debe ser < 1.")

    rng = random.Random(seed)
    texts = list(texts)  # copia
    rng.shuffle(texts)

    n = len(texts)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_texts = texts[:n_train]
    val_texts = texts[n_train:n_train + n_val]
    test_texts = texts[n_train + n_val:]

    return {"train": train_texts, "val": val_texts, "test": test_texts}


def _tokenize_function(examples, tokenizer: PreTrainedTokenizerBase, max_length: int):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",  
        max_length=max_length,
    )



def _add_labels(examples):
    
    examples["labels"] = examples["input_ids"].copy()
    return examples


def build_tokenized_datasets(
    splits: Dict[str, List[str]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
) -> DatasetDict:

    datasets_dict: Dict[str, Dataset] = {}
    for split_name, texts in splits.items():
        datasets_dict[split_name] = Dataset.from_dict({"text": texts})

    ds = DatasetDict(datasets_dict)


    ds = ds.map(
        lambda batch: _tokenize_function(batch, tokenizer, max_length),
        batched=True,
        remove_columns=["text"],
    )

 
    ds = ds.map(_add_labels, batched=True)

    return ds
