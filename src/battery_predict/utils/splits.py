from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random


@dataclass(frozen=True, slots=True)
class BatterySplit:
    train: tuple[Path, ...]
    val: tuple[Path, ...]


def split_battery_files(
    files: list[Path],
    *,
    seed: int,
    val_fraction: float,
) -> BatterySplit:
    if not files:
        raise ValueError("No battery files were found.")
    if val_fraction < 0 or val_fraction >= 1.0:
        raise ValueError("Validation fraction must be in the range [0, 1).")

    ordered = sorted(files)
    rng = random.Random(seed)
    rng.shuffle(ordered)

    n_total = len(ordered)
    n_val = max(1, round(n_total * val_fraction)) if val_fraction > 0 else 0
    if n_val >= n_total:
        n_val = n_total - 1

    val = tuple(ordered[:n_val])
    train = tuple(ordered[n_val:])
    if not train:
        raise ValueError(
            "Train split is empty; provide more files or a smaller validation fraction."
        )
    return BatterySplit(train=train, val=val)
