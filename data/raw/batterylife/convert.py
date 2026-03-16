"""Convert raw BatteryLife sodium-ion data into project .npy tensors.

Each output file represents one battery cell and is stored in `data/set` with a
4-character base62 filename derived from tensor contents. The payload tensor has
shape `(cycle, sample, channel)` where channel order is:
0 -> voltage, 1 -> current.

To address known raw-data issues seen during dataset inspection:
- Irregular sampling cadence is resampled to dt=1s via linear interpolation.
- Duplicate timestamps are merged by averaging values at the same timestamp.
- Non-finite rows are ignored.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RAW_DIR = Path(__file__).resolve().parent / "set" / "naion"
OUT_DIR = Path(__file__).resolve().parents[2] / "set"

BASE62_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
BASE = len(BASE62_ALPHABET)
HASH_LEN = 4
HASH_SPACE = BASE**HASH_LEN


@dataclass
class ConversionStats:
    files_seen: int = 0
    files_converted: int = 0
    files_reused: int = 0
    files_skipped_empty: int = 0


def encode_base62_fixed(value: int, width: int = HASH_LEN) -> str:
    chars = ["0"] * width
    for idx in range(width - 1, -1, -1):
        value, rem = divmod(value, BASE)
        chars[idx] = BASE62_ALPHABET[rem]
    return "".join(chars)


def merge_duplicate_timestamps(
    times: np.ndarray,
    voltage: np.ndarray,
    current: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(times)
    times_sorted = times[order]
    voltage_sorted = voltage[order]
    current_sorted = current[order]

    unique_times, inverse = np.unique(times_sorted, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)

    voltage_sum = np.bincount(inverse, weights=voltage_sorted)
    current_sum = np.bincount(inverse, weights=current_sorted)

    voltage_avg = voltage_sum / counts
    current_avg = current_sum / counts
    return unique_times, voltage_avg, current_avg


def cycle_to_uniform_signal(cycle: dict, dt_s: float = 1.0) -> np.ndarray | None:
    time_values = np.asarray(cycle.get("time_in_s") or [], dtype=np.float64)
    voltage_values = np.asarray(cycle.get("voltage_in_V") or [], dtype=np.float64)
    current_values = np.asarray(cycle.get("current_in_A") or [], dtype=np.float64)

    if not (len(time_values) and len(voltage_values) and len(current_values)):
        return None

    n = min(len(time_values), len(voltage_values), len(current_values))
    time_values = time_values[:n]
    voltage_values = voltage_values[:n]
    current_values = current_values[:n]

    finite_mask = (
        np.isfinite(time_values)
        & np.isfinite(voltage_values)
        & np.isfinite(current_values)
    )
    if finite_mask.sum() == 0:
        return None

    time_values = time_values[finite_mask]
    voltage_values = voltage_values[finite_mask]
    current_values = current_values[finite_mask]

    if len(time_values) == 1:
        sample = np.array([[voltage_values[0], current_values[0]]], dtype=np.float32)
        return sample

    time_values, voltage_values, current_values = merge_duplicate_timestamps(
        time_values,
        voltage_values,
        current_values,
    )

    if len(time_values) == 1:
        sample = np.array([[voltage_values[0], current_values[0]]], dtype=np.float32)
        return sample

    start = int(np.ceil(time_values[0]))
    end = int(np.floor(time_values[-1]))

    if start > end:
        anchor = int(round(float(time_values[0])))
        grid = np.array([anchor], dtype=np.float64)
    else:
        grid = np.arange(start, end + dt_s, dt_s, dtype=np.float64)

    voltage_interp = np.interp(grid, time_values, voltage_values)
    current_interp = np.interp(grid, time_values, current_values)
    return np.stack([voltage_interp, current_interp], axis=-1).astype(np.float32)


def battery_to_tensor(battery: dict, dt_s: float = 1.0) -> np.ndarray | None:
    cycles = battery.get("cycle_data") or []
    cycle_arrays: list[np.ndarray] = []

    for cycle in cycles:
        arr = cycle_to_uniform_signal(cycle, dt_s=dt_s)
        if arr is not None and arr.shape[0] > 0:
            cycle_arrays.append(arr)

    if not cycle_arrays:
        return None

    max_samples = max(arr.shape[0] for arr in cycle_arrays)
    tensor = np.full((len(cycle_arrays), max_samples, 2), np.nan, dtype=np.float32)

    for i, arr in enumerate(cycle_arrays):
        tensor[i, : arr.shape[0], :] = arr
    return tensor


def tensors_equal(existing_path: Path, tensor: np.ndarray) -> bool:
    try:
        existing = np.load(existing_path, allow_pickle=False)
    except Exception:
        return False

    if existing.shape != tensor.shape or existing.dtype != tensor.dtype:
        return False
    return np.array_equal(existing, tensor, equal_nan=True)


def choose_output_path(tensor: np.ndarray, out_dir: Path) -> tuple[Path, bool]:
    digest = hashlib.sha256(tensor.tobytes(order="C")).digest()
    seed = int.from_bytes(digest, byteorder="big") % HASH_SPACE

    for offset in range(HASH_SPACE):
        idx = (seed + offset) % HASH_SPACE
        name = encode_base62_fixed(idx)
        candidate = out_dir / f"{name}.npy"

        if not candidate.exists():
            return candidate, False

        if tensors_equal(candidate, tensor):
            return candidate, True

    raise RuntimeError("Hash space exhausted: could not place tensor.")


def convert(
    raw_dir: Path = RAW_DIR, out_dir: Path = OUT_DIR, dt_s: float = 1.0
) -> ConversionStats:
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = ConversionStats()
    for src in sorted(raw_dir.glob("*.pkl")):
        stats.files_seen += 1
        with src.open("rb") as f:
            battery = pickle.load(f)

        tensor = battery_to_tensor(battery, dt_s=dt_s)
        if tensor is None:
            stats.files_skipped_empty += 1
            print(f"[skip] {src.name}: no usable cycle samples")
            continue

        dst, existed_same = choose_output_path(tensor, out_dir)
        if existed_same:
            stats.files_reused += 1
            print(
                f"[keep] {src.name} -> {dst.name} (identical content already present)"
            )
            continue

        np.save(dst, tensor, allow_pickle=False)
        stats.files_converted += 1
        print(f"[save] {src.name} -> {dst.name} shape={tensor.shape}")

    return stats


def main() -> None:
    stats = convert()
    print(
        "\nSummary:"
        f" seen={stats.files_seen},"
        f" saved={stats.files_converted},"
        f" reused={stats.files_reused},"
        f" skipped_empty={stats.files_skipped_empty}"
    )


if __name__ == "__main__":
    main()
