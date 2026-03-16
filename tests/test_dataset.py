from __future__ import annotations

import numpy as np

from battery_predict.data.dataset import BatteryDataModule, collate_cycle_windows
from battery_predict.training.config import DataConfig


def _make_battery(num_cycles: int, valid_len: int) -> np.ndarray:
    battery = np.full((num_cycles, valid_len + 2, 2), np.nan, dtype=np.float32)
    for cycle_idx in range(num_cycles):
        battery[cycle_idx, :valid_len, 0] = np.linspace(
            4.2, 3.0, valid_len, dtype=np.float32
        )
        battery[cycle_idx, :valid_len, 1] = np.linspace(
            -2.0, -1.0, valid_len, dtype=np.float32
        )
    return battery


def test_collate_cycle_windows_pads_signals_and_sequences() -> None:
    batch = [
        {
            "battery_id": "a",
            "signals": (
                np.ones((3, 2), dtype=np.float32),
                np.ones((2, 2), dtype=np.float32),
            ),
            "signal_masks": (np.ones(3, dtype=bool), np.ones(2, dtype=bool)),
            "capacities_ah": np.array([0.2, 0.3], dtype=np.float32),
            "capacity_valid": np.array([True, True]),
            "cycle_indices": np.array([0, 1], dtype=np.int64),
        },
        {
            "battery_id": "b",
            "signals": (np.ones((4, 2), dtype=np.float32),),
            "signal_masks": (np.ones(4, dtype=bool),),
            "capacities_ah": np.array([0.4], dtype=np.float32),
            "capacity_valid": np.array([True]),
            "cycle_indices": np.array([3], dtype=np.int64),
        },
    ]

    result = collate_cycle_windows(batch)
    assert result["signals"].shape == (2, 2, 4, 2)
    assert result["signal_mask"].shape == (2, 2, 4)
    assert result["sequence_mask"].tolist() == [[True, True], [True, False]]


def test_datamodule_uses_epoch_samples_in_train_sampler(tmp_path) -> None:
    dataset_dir = tmp_path / "set"
    dataset_dir.mkdir()
    for idx in range(10):
        np.save(
            dataset_dir / f"cell_{idx:03d}.npy", _make_battery(6, 8), allow_pickle=False
        )

    config = DataConfig(
        dataset_dir=str(dataset_dir),
        cycle_window=4,
        epoch_samples=5,
        val_epoch_samples=3,
        train_batch_size=2,
        eval_batch_size=2,
    )
    datamodule = BatteryDataModule(config)
    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    assert train_loader.sampler is not None
    assert getattr(train_loader.sampler, "num_samples", None) == 5
    assert val_loader.sampler is not None
    assert getattr(val_loader.sampler, "num_samples", None) == 3
