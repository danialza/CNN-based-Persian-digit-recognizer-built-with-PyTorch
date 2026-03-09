from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

DIGIT_TO_PERSIAN = {
    0: "\u06f0",
    1: "\u06f1",
    2: "\u06f2",
    3: "\u06f3",
    4: "\u06f4",
    5: "\u06f5",
    6: "\u06f6",
    7: "\u06f7",
    8: "\u06f8",
    9: "\u06f9",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def labels_with_persian(class_names: Iterable[str]) -> list[str]:
    labels: list[str] = []
    for name in class_names:
        try:
            digit = int(name)
        except ValueError:
            labels.append(name)
            continue
        persian = DIGIT_TO_PERSIAN.get(digit)
        labels.append(f"{name} ({persian})" if persian else name)
    return labels
