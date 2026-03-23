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

# Progress bar for dataset construction
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kwargs):
        return x


@dataclass(slots=True)
class BatteryRecord:
    path: Path
    valid_cycle_indices: np.ndarray
    capacities_ah: np.ndarray
    capacity_valid: np.ndarray

    @property
    def num_cycles(self) -> int:
        return int(self.valid_cycle_indices.shape[0])


def _load_battery_record(path: Path, config: DataConfig) -> BatteryRecord:
    # Memory-map large .npy files to avoid loading the full tensor in RAM.
    array = np.load(path, allow_pickle=False, mmap_mode="r")
    if array.ndim != 3 or array.shape[-1] != 2:
        raise ValueError(f"Battery tensor {path} must have shape (cycle, sample, 2).")

    valid_cycle_indices: list[int] = []
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
        valid_cycle_indices.append(cycle_idx)
        capacities.append(capacity_ah)
        capacity_valid.append(is_valid_capacity)

    if not valid_cycle_indices:
        raise ValueError(f"Battery tensor {path} has no valid cycles.")

    valid_cycle_indices_array = np.asarray(valid_cycle_indices, dtype=np.int64)
    capacity_valid_array = np.asarray(capacity_valid, dtype=bool)
    if config.drop_cycles_without_discharge:
        keep = capacity_valid_array
        valid_cycle_indices_array = valid_cycle_indices_array[keep]
        capacities = [
            value for value, include in zip(capacities, keep, strict=True) if include
        ]
        capacity_valid_array = np.ones(valid_cycle_indices_array.shape[0], dtype=bool)

    return BatteryRecord(
        path=path,
        valid_cycle_indices=valid_cycle_indices_array,
        capacities_ah=np.asarray(capacities, dtype=np.float32),
        capacity_valid=capacity_valid_array.astype(bool, copy=False),
    )


def _count_windows(record: BatteryRecord, config: DataConfig) -> int:
    # Require a full context window and at least min_pred_seq_len target cycles.
    required_cycles = config.cycle_window + config.min_pred_seq_len
    if record.num_cycles < required_cycles:
        return 0

    return record.num_cycles - required_cycles + 1


class BatteryWindowDataset(Dataset[dict[str, Any]]):
    def __init__(self, files: tuple[Path, ...], config: DataConfig):
        self.config = config
        self.files = tuple(sorted(files))
        self._cache: dict[Path, BatteryRecord] = {}
        self._records: list[BatteryRecord] = []
        self._window_offsets = np.asarray([0], dtype=np.int64)
        self._total_windows = 0
        self._active_array_path: Path | None = None
        self._active_array: np.ndarray | None = None

        counts: list[int] = []
        for path in tqdm(self.files, desc="Indexing battery files", unit="file"):
            record = self._get_record(path)
            self._records.append(record)
            counts.append(_count_windows(record, config))

        if counts:
            counts_array = np.asarray(counts, dtype=np.int64)
            self._window_offsets = np.concatenate(
                (
                    np.asarray([0], dtype=np.int64),
                    np.cumsum(counts_array, dtype=np.int64),
                )
            )
            self._total_windows = int(self._window_offsets[-1])

        if self._total_windows <= 0:
            raise ValueError(
                "No valid cycle windows were constructed from the selected files."
            )

    def _get_record(self, path: Path) -> BatteryRecord:
        if path not in self._cache:
            self._cache[path] = _load_battery_record(path, self.config)
        return self._cache[path]

    def _get_array(self, path: Path) -> np.ndarray:
        if self._active_array_path != path or self._active_array is None:
            self._active_array = np.load(path, allow_pickle=False, mmap_mode="r")
            self._active_array_path = path
        assert self._active_array is not None
        return self._active_array

    def _resolve_index(self, index: int) -> tuple[BatteryRecord, int]:
        if index < 0:
            index += self._total_windows
        if index < 0 or index >= self._total_windows:
            raise IndexError("BatteryWindowDataset index out of range")

        file_idx = int(np.searchsorted(self._window_offsets, index, side="right") - 1)
        start = int(index - self._window_offsets[file_idx])
        return self._records[file_idx], start

    @property
    def windows(self) -> tuple[tuple[Path, int], ...]:
        flat_windows: list[tuple[Path, int]] = []
        for file_idx, record in enumerate(self._records):
            start = int(self._window_offsets[file_idx])
            end = int(self._window_offsets[file_idx + 1])
            flat_windows.extend((record.path, offset) for offset in range(end - start))
        return tuple(flat_windows)

    def __len__(self) -> int:
        return self._total_windows

    def __getitem__(self, index: int) -> dict[str, Any]:
        record, start = self._resolve_index(index)
        path = record.path
        array = self._get_array(path)

        context_end = start + self.config.cycle_window

        source_cycle_indices = record.valid_cycle_indices[start:context_end]
        cycles = []
        for source_cycle_idx in source_cycle_indices:
            cycle = array[int(source_cycle_idx)]
            valid_mask = compute_valid_cycle_mask(cycle)
            valid_len = int(valid_mask.sum())
            cycles.append(cycle[:valid_len].astype(np.float32, copy=False))
        cycles_tuple = tuple(cycles)

        capacities = record.capacities_ah[start:context_end]
        capacity_valid = record.capacity_valid[start:context_end]

        target_start = context_end
        max_pred = getattr(self.config, "max_pred_seq_len", None)
        target_caps = record.capacities_ah[target_start:].astype(np.float32, copy=False)
        target_valid_mask = record.capacity_valid[target_start:].astype(
            bool, copy=False
        )
        if max_pred is not None:
            target_caps = target_caps[:max_pred]
            target_valid_mask = target_valid_mask[:max_pred]

        return {
            "battery_id": path.stem,
            "signals": cycles_tuple,
            "signal_masks": tuple(np.ones(len(c), dtype=bool) for c in cycles_tuple),
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
            # Signals can come from read-only mmap views; force writable copy first.
            signals[batch_idx, cycle_idx, :valid_len, :] = torch.from_numpy(
                np.array(signal, copy=True)
            )
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

    _setup_call_count = 0

    def setup(self, stage: str | None = None) -> None:
        type(self)._setup_call_count += 1
        if type(self)._setup_call_count > 1:
            print(
                "[WARNING] BatteryDataModule.setup() called more than once. This may cause double dataset construction."
            )
        files = sorted(self.dataset_dir.glob("*.npy"))
        print(f"[BatteryDataModule] Scanning {len(files)} .npy files for split...")
        self.split = split_battery_files(
            files,
            seed=self.config.split_seed,
            val_fraction=self.config.val_fraction,
        )

        if stage in (None, "fit"):
            print(
                f"[BatteryDataModule] Building train dataset ({len(self.split.train)} files)..."
            )
            self.train_dataset = BatteryWindowDataset(self.split.train, self.config)
            print(
                f"[BatteryDataModule] Building val dataset ({len(self.split.val)} files)..."
            )
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
        val_samples = getattr(self.config, "utilize_val_epoch_windows", None)
        if val_samples is not None:
            sampler = RandomSampler(
                self.val_dataset,
                replacement=val_samples > len(self.val_dataset),
                num_samples=val_samples,
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
