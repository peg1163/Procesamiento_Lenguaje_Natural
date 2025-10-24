import os, json, hashlib, random
from pathlib import Path
from typing import Any
import numpy as np
import torch

DEF_OUT = Path(os.getenv("OUT", "out"))

def set_seed(seed: int = 1337) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def device_select(kind: str = "auto") -> torch.device:
    if kind == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(kind)

def sha256_bytes(b: bytes, salt: str = "") -> str:
    return hashlib.sha256(b + salt.encode()).hexdigest()

def save_json(obj: Any, path: os.PathLike | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))
