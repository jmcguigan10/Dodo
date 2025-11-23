import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(state: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Optional[str], map_location: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    chk_path = Path(path)
    if not chk_path.exists():
        return None
    return torch.load(chk_path, map_location=map_location)
