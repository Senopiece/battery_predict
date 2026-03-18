"""Dataset analysis utilities shared between notebooks and training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def valid_cycle_count_for_file(path: Path, config: dict[str, Any]) -> int:
    """Count the number of valid cycles in a battery file.

    A cycle is valid if it meets the discharge capacity threshold.

    Args:
        path: Path to the .npy battery file.
        config: Data config dict with keys:
            - min_discharge_capacity_ah: threshold for valid discharge
            - dt_seconds: sampling interval
            - drop_cycles_without_discharge: whether to filter cycles

    Returns:
        Number of valid cycles in the file.
    """
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 3 or arr.shape[-1] != 2:
        return 0

    valid_cycles = 0
    min_q = float(config.get("min_discharge_capacity_ah", 1e-6))
    dt_seconds = float(config.get("dt_seconds", 1.0))

    for cycle_idx in range(arr.shape[0]):
        cycle = arr[cycle_idx]
        valid_mask = np.isfinite(cycle[:, 0]) & np.isfinite(cycle[:, 1])
        if valid_mask.sum() <= 0:
            continue
        current = cycle[valid_mask, 1]
        discharge_capacity_ah = float(
            np.clip(-current, 0.0, None).sum() * dt_seconds / 3600.0
        )

        drop = config.get("drop_cycles_without_discharge", True)
        if drop:
            if discharge_capacity_ah >= min_q:
                valid_cycles += 1
        else:
            valid_cycles += 1

    return valid_cycles


def count_windows(
    file_group: list[Path],
    config: dict[str, Any],
) -> tuple[int, dict[str, int]]:
    """Count total and per-file window counts for a file group.

    A window is a contiguous sequence of valid cycles.

    Args:
        file_group: List of battery file paths.
        config: Data config dict with keys:
            - cycle_window: length of each window
            - min_observed_cycles: minimum cycles to form a window
            - (plus keys needed by valid_cycle_count_for_file)

    Returns:
        Tuple of (total_windows, per_file_window_counts).
    """
    total = 0
    per_file = {}
    cycle_window = int(config.get("cycle_window", 12))
    min_observed = int(config.get("min_observed_cycles", 2))

    for path in file_group:
        n_cycles = valid_cycle_count_for_file(path, config)
        if n_cycles < min_observed:
            windows = 0
        elif n_cycles <= cycle_window:
            windows = 1
        else:
            windows = n_cycles - cycle_window + 1
        per_file[path.name] = windows
        total += windows

    return total, per_file


def analyze_signal_lengths(
    files: list[Path],
    config: dict[str, Any],
    downsampled: bool = False,
) -> dict[str, Any]:
    """Analyze signal length statistics across all files.

    Args:
        files: List of battery file paths.
        config: Config dict (needs encoder config for conv downsampling).
        downsampled: If True, apply Conv1D downsampling to estimate downsampled lengths.

    Returns:
        Dict with keys: min, median, p95, p99, p99_9, max.
    """
    from statistics import median as stat_median
    import math

    all_lengths = []

    for path in files:
        arr = np.load(path, allow_pickle=False)
        if arr.ndim != 3 or arr.shape[-1] != 2:
            continue

        for cycle_idx in range(arr.shape[0]):
            cycle = arr[cycle_idx]
            valid_mask = np.isfinite(cycle[:, 0]) & np.isfinite(cycle[:, 1])
            valid_len = int(valid_mask.sum())
            all_lengths.append(valid_len)

    if not all_lengths:
        return {
            "min": 0,
            "median": 0,
            "p95": 0,
            "p99": 0,
            "p99_9": 0,
            "max": 0,
        }

    if downsampled and "encoder" in config:
        # Apply Conv1D downsampling
        conv_kernels = config["encoder"].get("conv_kernels", [5, 3])
        conv_strides = config["encoder"].get("conv_strides", [1, 2])

        def conv1d_out_len(
            length: int,
            kernel: int,
            stride: int,
            padding: int = None,
            dilation: int = 1,
        ) -> int:
            if padding is None:
                padding = kernel // 2
            return math.floor(
                (length + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
            )

        downsampled_lengths = []
        for length in all_lengths:
            out = int(length)
            for kernel, stride in zip(conv_kernels, conv_strides, strict=True):
                out = conv1d_out_len(out, kernel=kernel, stride=stride)
            downsampled_lengths.append(max(out, 0))
        all_lengths = downsampled_lengths

    lengths_arr = np.asarray(all_lengths, dtype=np.int64)
    return {
        "min": int(lengths_arr.min()),
        "median": int(np.median(lengths_arr)),
        "p95": int(np.percentile(lengths_arr, 95)),
        "p99": int(np.percentile(lengths_arr, 99)),
        "p99_9": int(np.percentile(lengths_arr, 99.9)),
        "max": int(lengths_arr.max()),
    }
