import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch
import yaml


LOGGER = logging.getLogger("pc3")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)


def load_config(path: str) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró config: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    LOGGER.info(f"Seed fijada a {seed}")


def get_device(preferred: str = "cuda") -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    step: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "step": step,
        "model_state": model.state_dict(),
    }
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state"] = scheduler.state_dict()
    if extra:
        state["extra"] = extra

    torch.save(state, path)
    LOGGER.info(f"Checkpoint guardado en {path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    map_location: Optional[str] = None,
) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró checkpoint: {path}")

    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model_state"])

    if optimizer is not None and "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])
    if scheduler is not None and "scheduler_state" in state:
        scheduler.load_state_dict(state["scheduler_state"])

    LOGGER.info(f"Checkpoint cargado desde {path} (step={state.get('step', 0)})")
    return state



def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_json(obj: Any, path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_exp(x: float) -> float:
    
    try:
        return float(math.exp(min(x, 50)))
    except OverflowError:
        return float("inf")
