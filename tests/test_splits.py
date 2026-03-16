from __future__ import annotations

from pathlib import Path

from battery_predict.utils.splits import split_battery_files


def test_split_battery_files_is_reproducible() -> None:
    files = [Path(f"cell_{idx:03d}.npy") for idx in range(20)]

    split_a = split_battery_files(files, seed=13, val_fraction=0.2, test_fraction=0.1)
    split_b = split_battery_files(files, seed=13, val_fraction=0.2, test_fraction=0.1)
    split_c = split_battery_files(files, seed=99, val_fraction=0.2, test_fraction=0.1)

    assert split_a == split_b
    assert split_a != split_c
    assert len(split_a.train) + len(split_a.val) + len(split_a.test) == len(files)
