from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from battery_predict.training.config import DataConfig
from battery_predict.utils.capacity import (
    compute_discharge_capacity_ah,
    compute_valid_cycle_mask,
)
from battery_predict.utils.splits import BatterySplit, split_battery_files


@dataclass(slots=True)
class BatteryRecord:
    path: Path
    cycles: tuple[np.ndarray, ...]
    capacities_ah: np.ndarray
    capacity_valid: np.ndarray

    @property
    def num_cycles(self) -> int:
        return len(self.cycles)


def _load_battery_record(path: Path, config: DataConfig) -> BatteryRecord:
    array = np.load(path, allow_pickle=False)
    if array.ndim != 3 or array.shape[-1] != 2:
        raise ValueError(f"Battery tensor {path} must have shape (cycle, sample, 2).")

    cycles: list[np.ndarray] = []
    capacities: list[float] = []
    capacity_valid: list[bool] = []

    for cycle_idx in range(array.shape[0]):
        cycle = array[cycle_idx]
        valid_mask = compute_valid_cycle_mask(cycle)
        valid_len = int(valid_mask.sum())
        if valid_len <= 0:
            continue

        trimmed = cycle[:valid_len].astype(np.float32, copy=False)
        capacity_ah, is_valid_capacity = compute_discharge_capacity_ah(
            trimmed,
            dt_seconds=config.dt_seconds,
            min_capacity_ah=config.min_discharge_capacity_ah,
        )
        cycles.append(trimmed)
        capacities.append(capacity_ah)
        capacity_valid.append(is_valid_capacity)

    if not cycles:
        raise ValueError(f"Battery tensor {path} has no valid cycles.")

    capacity_valid_array = np.asarray(capacity_valid, dtype=bool)
    if config.drop_cycles_without_discharge:
        keep = capacity_valid_array
        cycles = [cycle for cycle, include in zip(cycles, keep, strict=True) if include]
        capacities = [
            value for value, include in zip(capacities, keep, strict=True) if include
        ]
        capacity_valid_array = np.ones(len(cycles), dtype=bool)

    return BatteryRecord(
        path=path,
        cycles=tuple(cycles),
        capacities_ah=np.asarray(capacities, dtype=np.float32),
        capacity_valid=capacity_valid_array.astype(bool, copy=False),
    )


def _build_window_index(
    record: BatteryRecord, config: DataConfig
) -> list[tuple[Path, int]]:
    # Require a full context window and at least min_pred_seq_len target cycles.
    required_cycles = config.cycle_window + config.min_pred_seq_len
    if record.num_cycles < required_cycles:
        return []

    max_start = record.num_cycles - required_cycles + 1
    return [(record.path, start) for start in range(max_start)]


class BatteryWindowDataset(Dataset[dict[str, Any]]):
    def __init__(self, files: tuple[Path, ...], config: DataConfig):
        self.config = config
        self.files = tuple(sorted(files))
        self._cache: dict[Path, BatteryRecord] = {}
        self._window_index: list[tuple[Path, int]] = []

        for path in self.files:
            record = self._get_record(path)
            self._window_index.extend(_build_window_index(record, config))

        if not self._window_index:
            raise ValueError(
                "No valid cycle windows were constructed from the selected files."
            )

    def _get_record(self, path: Path) -> BatteryRecord:
        if path not in self._cache:
            self._cache[path] = _load_battery_record(path, self.config)
        return self._cache[path]

    @property
    def windows(self) -> tuple[tuple[Path, int], ...]:
        return tuple(self._window_index)

    def __len__(self) -> int:
        return len(self._window_index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, start = self._window_index[index]
        record = self._get_record(path)

        context_end = start + self.config.cycle_window

        cycles = record.cycles[start:context_end]
        capacities = record.capacities_ah[start:context_end]
        capacity_valid = record.capacity_valid[start:context_end]

        target_start = context_end
        target_caps = record.capacities_ah[target_start:].astype(np.float32, copy=False)
        target_valid_mask = record.capacity_valid[target_start:].astype(
            bool, copy=False
        )

        return {
            "battery_id": path.stem,
            "signals": cycles,
            "signal_masks": tuple(np.ones(len(c), dtype=bool) for c in cycles),
            "capacities_ah": capacities.astype(np.float32, copy=False),
            "capacity_valid": capacity_valid.astype(bool, copy=False),
            "target_capacities_ah": target_caps,
            "target_capacity_valid": target_valid_mask,
            "cycle_indices": np.arange(start, context_end, dtype=np.int64),
        }


def collate_cycle_windows(batch: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(batch)
    max_cycles = max(len(item["signals"]) for item in batch)
    max_samples = max(signal.shape[0] for item in batch for signal in item["signals"])
    max_target_len = max(item["target_capacities_ah"].shape[0] for item in batch)

    signals = torch.zeros((batch_size, max_cycles, max_samples, 2), dtype=torch.float32)
    signal_mask = torch.zeros((batch_size, max_cycles, max_samples), dtype=torch.bool)
    sequence_mask = torch.zeros((batch_size, max_cycles), dtype=torch.bool)
    capacities_ah = torch.zeros((batch_size, max_cycles), dtype=torch.float32)
    capacity_valid = torch.zeros((batch_size, max_cycles), dtype=torch.bool)
    cycle_indices = torch.full((batch_size, max_cycles), -1, dtype=torch.long)
    target_capacities_ah = torch.zeros(
        (batch_size, max_target_len), dtype=torch.float32
    )
    target_capacity_valid = torch.zeros((batch_size, max_target_len), dtype=torch.bool)
    battery_ids: list[str] = []

    for batch_idx, item in enumerate(batch):
        battery_ids.append(item["battery_id"])
        num_cycles = len(item["signals"])
        sequence_mask[batch_idx, :num_cycles] = True
        capacities_ah[batch_idx, :num_cycles] = torch.from_numpy(item["capacities_ah"])
        capacity_valid[batch_idx, :num_cycles] = torch.from_numpy(
            item["capacity_valid"]
        )
        cycle_indices[batch_idx, :num_cycles] = torch.from_numpy(item["cycle_indices"])
        target_len = item["target_capacities_ah"].shape[0]
        target_capacities_ah[batch_idx, :target_len] = torch.from_numpy(
            item["target_capacities_ah"]
        )
        target_capacity_valid[batch_idx, :target_len] = torch.from_numpy(
            item["target_capacity_valid"]
        )

        for cycle_idx, signal in enumerate(item["signals"]):
            valid_len = signal.shape[0]
            signals[batch_idx, cycle_idx, :valid_len, :] = torch.from_numpy(signal)
            signal_mask[batch_idx, cycle_idx, :valid_len] = True

    return {
        "battery_ids": battery_ids,
        "signals": signals,
        "signal_mask": signal_mask,
        "sequence_mask": sequence_mask,
        "capacities_ah": capacities_ah,
        "capacity_valid": capacity_valid,
        "target_capacities_ah": target_capacities_ah,
        "target_capacity_valid": target_capacity_valid,
        "cycle_indices": cycle_indices,
    }


class BatteryDataModule(L.LightningDataModule):
    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config
        self.dataset_dir = Path(config.dataset_dir)
        self.split: BatterySplit | None = None
        self.train_dataset: BatteryWindowDataset | None = None
        self.val_dataset: BatteryWindowDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        files = sorted(self.dataset_dir.glob("*.npy"))
        self.split = split_battery_files(
            files,
            seed=self.config.split_seed,
            val_fraction=self.config.val_fraction,
        )

        if stage in (None, "fit"):
            self.train_dataset = BatteryWindowDataset(self.split.train, self.config)
            self.val_dataset = BatteryWindowDataset(self.split.val, self.config)

    def train_dataloader(self) -> DataLoader[dict[str, Any]]:
        if self.train_dataset is None:
            raise RuntimeError(
                "DataModule.setup('fit') must be called before train_dataloader()."
            )

        sampler = None
        shuffle = False
        if self.config.utilize_epoch_windows is not None:
            sampler = RandomSampler(
                self.train_dataset,
                replacement=self.config.utilize_epoch_windows > len(self.train_dataset),
                num_samples=self.config.utilize_epoch_windows,
            )
        else:
            shuffle = True

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers
            and self.config.num_workers > 0,
            collate_fn=collate_cycle_windows,
        )

    def val_dataloader(self) -> DataLoader[dict[str, Any]]:
        if self.val_dataset is None:
            raise RuntimeError(
                "DataModule.setup('fit') must be called before val_dataloader()."
            )
        sampler = None
        if getattr(self.config, "utilize_val_epoch_windows", None) is not None:
            sampler = RandomSampler(
                self.val_dataset,
                replacement=self.config.utilize_val_epoch_windows
                > len(self.val_dataset),
                num_samples=self.config.utilize_val_epoch_windows,
            )
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.val_batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers
            and self.config.num_workers > 0,
            collate_fn=collate_cycle_windows,
        )
