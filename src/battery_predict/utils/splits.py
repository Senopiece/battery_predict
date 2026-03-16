from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random


@dataclass(frozen=True, slots=True)
class BatterySplit:
    train: tuple[Path, ...]
    val: tuple[Path, ...]
    test: tuple[Path, ...]


def split_battery_files(
    files: list[Path],
    *,
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> BatterySplit:
    if not files:
        raise ValueError("No battery files were found.")
    if val_fraction < 0 or test_fraction < 0 or (val_fraction + test_fraction) >= 1.0:
        raise ValueError(
            "Validation and test fractions must be non-negative and sum to < 1."
        )

    ordered = sorted(files)
    rng = random.Random(seed)
    rng.shuffle(ordered)

    n_total = len(ordered)
    n_test = max(1, round(n_total * test_fraction)) if test_fraction > 0 else 0
    n_val = max(1, round(n_total * val_fraction)) if val_fraction > 0 else 0
    if n_val + n_test >= n_total:
        overflow = n_val + n_test - (n_total - 1)
        if overflow > 0:
            if n_val >= n_test and n_val > 1:
                n_val -= overflow
            elif n_test > 1:
                n_test -= overflow

    test = tuple(ordered[:n_test])
    val = tuple(ordered[n_test : n_test + n_val])
    train = tuple(ordered[n_test + n_val :])
    if not train:
        raise ValueError(
            "Train split is empty; provide more files or smaller holdout fractions."
        )
    return BatterySplit(train=train, val=val, test=test)
