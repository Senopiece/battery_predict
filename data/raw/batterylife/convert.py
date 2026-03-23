"""Convert raw BatteryLife sodium-ion data into project artifacts.

This script converts two splits:
- Training split: `data/raw/batterylife/set/naion/*.pkl` -> `data/set/*.npy`
- Heldout split: `data/raw/batterylife/heldout/naion/*.pkl` ->
    `data/set/heldout/*.jsonl`

Both outputs use deterministic 4-character base62 names derived from payload
content. Heldout JSONL files are written as one JSON object per cycle:
`{"V": [...], "A": [...]}` where arrays are 1 Hz samples.

To address known raw-data issues seen during dataset inspection:
- Irregular sampling cadence is resampled to dt=1s via linear interpolation.
- Duplicate timestamps are merged by averaging values at the same timestamp.
- Non-finite rows are ignored.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

RAW_DIR = Path(__file__).resolve().parent / "set" / "liion"
OUT_DIR = Path(__file__).resolve().parents[2] / "set"
RAW_HELDOUT_DIR = Path(__file__).resolve().parent / "heldout" / "liion"
OUT_HELDOUT_DIR = OUT_DIR / "heldout"

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


def load_battery(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".pkl":
        with path.open("rb") as handle:
            return pickle.load(handle)
    raise ValueError(f"Unsupported source extension: {path.suffix}")


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


def cycle_to_uniform_trace(
    cycle: dict,
    dt_s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
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
        grid = np.array([float(round(float(time_values[0])))], dtype=np.float64)
        voltage = np.array([voltage_values[0]], dtype=np.float32)
        current = np.array([current_values[0]], dtype=np.float32)
        return grid, voltage, current

    time_values, voltage_values, current_values = merge_duplicate_timestamps(
        time_values,
        voltage_values,
        current_values,
    )

    if len(time_values) == 1:
        grid = np.array([float(round(float(time_values[0])))], dtype=np.float64)
        voltage = np.array([voltage_values[0]], dtype=np.float32)
        current = np.array([current_values[0]], dtype=np.float32)
        return grid, voltage, current

    start = int(np.ceil(time_values[0]))
    end = int(np.floor(time_values[-1]))

    if start > end:
        anchor = int(round(float(time_values[0])))
        grid = np.array([anchor], dtype=np.float64)
    else:
        grid = np.arange(start, end + dt_s, dt_s, dtype=np.float64)

    voltage_interp = np.interp(grid, time_values, voltage_values).astype(np.float32)
    current_interp = np.interp(grid, time_values, current_values).astype(np.float32)
    return grid, voltage_interp, current_interp


def cycle_to_uniform_signal(cycle: dict, dt_s: float = 1.0) -> np.ndarray | None:
    trace = cycle_to_uniform_trace(cycle, dt_s=dt_s)
    if trace is None:
        return None
    _, voltage_interp, current_interp = trace
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


def jsonl_line(record: dict[str, list[float]]) -> str:
    return json.dumps(record, separators=(",", ":"), ensure_ascii=False)


def records_payload_bytes(records: list[dict[str, list[float]]]) -> bytes:
    text = "\n".join(jsonl_line(record) for record in records)
    return text.encode("utf-8")


def write_jsonl_records(path: Path, records: list[dict[str, list[float]]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(jsonl_line(record))
            handle.write("\n")


def jsonl_records_equal(
    existing_path: Path,
    records: list[dict[str, list[float]]],
) -> bool:
    try:
        with existing_path.open("r", encoding="utf-8") as handle:
            existing = [json.loads(line) for line in handle if line.strip()]
    except Exception:
        return False
    return existing == records


def choose_output_path(
    *,
    payload_bytes: bytes,
    out_dir: Path,
    suffix: str,
    is_same_payload: Callable[[Path], bool],
) -> tuple[Path, bool]:
    digest = hashlib.sha256(payload_bytes).digest()
    seed = int.from_bytes(digest, byteorder="big") % HASH_SPACE

    for offset in range(HASH_SPACE):
        idx = (seed + offset) % HASH_SPACE
        name = encode_base62_fixed(idx)
        candidate = out_dir / f"{name}{suffix}"

        if not candidate.exists():
            return candidate, False

        if is_same_payload(candidate):
            return candidate, True

    raise RuntimeError("Hash space exhausted: could not place tensor.")


def battery_to_cycle_records(
    battery: dict,
    dt_s: float = 1.0,
) -> list[dict[str, list[float]]] | None:
    cycles = battery.get("cycle_data") or []

    records: list[dict[str, list[float]]] = []

    for cycle in cycles:
        trace = cycle_to_uniform_trace(cycle, dt_s=dt_s)
        if trace is None:
            continue
        _, voltage, current = trace
        if voltage.size == 0:
            continue
        records.append(
            {
                "V": voltage.astype(np.float32).tolist(),
                "A": current.astype(np.float32).tolist(),
            }
        )

    if not records:
        return None

    return records


def convert_split(
    *,
    split_name: str,
    raw_dir: Path,
    out_dir: Path,
    patterns: tuple[str, ...],
    mode: str,
    dt_s: float = 1.0,
) -> ConversionStats:
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = ConversionStats()
    sources: list[Path] = []
    for pattern in patterns:
        sources.extend(raw_dir.glob(pattern))

    seen_paths: set[Path] = set()
    for src in sorted(sources):
        if src in seen_paths:
            continue
        seen_paths.add(src)
        stats.files_seen += 1
        battery = load_battery(src)

        if mode == "npy":
            tensor = battery_to_tensor(battery, dt_s=dt_s)
            if tensor is None:
                stats.files_skipped_empty += 1
                print(f"[{split_name}][skip] {src.name}: no usable cycle samples")
                continue

            dst, existed_same = choose_output_path(
                payload_bytes=tensor.tobytes(order="C"),
                out_dir=out_dir,
                suffix=".npy",
                is_same_payload=lambda candidate: tensors_equal(candidate, tensor),
            )
            if existed_same:
                stats.files_reused += 1
                print(
                    f"[{split_name}][keep] {src.name} -> {dst.name} "
                    "(identical content already present)"
                )
                continue

            np.save(dst, tensor, allow_pickle=False)
            stats.files_converted += 1
            print(f"[{split_name}][save] {src.name} -> {dst.name} shape={tensor.shape}")
            continue

        if mode == "jsonl":
            records = battery_to_cycle_records(battery, dt_s=dt_s)
            if records is None:
                stats.files_skipped_empty += 1
                print(f"[{split_name}][skip] {src.name}: no usable cycle samples")
                continue

            dst, existed_same = choose_output_path(
                payload_bytes=records_payload_bytes(records),
                out_dir=out_dir,
                suffix=".jsonl",
                is_same_payload=lambda candidate: jsonl_records_equal(
                    candidate, records
                ),
            )
            if existed_same:
                stats.files_reused += 1
                print(
                    f"[{split_name}][keep] {src.name} -> {dst.name} "
                    "(identical content already present)"
                )
                continue

            write_jsonl_records(dst, records)
            stats.files_converted += 1
            print(
                f"[{split_name}][save] {src.name} -> {dst.name} "
                f"cycles={len(records)}"
            )
            continue

        raise ValueError(f"Unsupported conversion mode: {mode}")

    return stats


def convert(
    raw_dir: Path = RAW_DIR,
    out_dir: Path = OUT_DIR,
    raw_heldout_dir: Path = RAW_HELDOUT_DIR,
    out_heldout_dir: Path = OUT_HELDOUT_DIR,
    dt_s: float = 1.0,
) -> tuple[ConversionStats, ConversionStats]:
    set_stats = convert_split(
        split_name="set",
        raw_dir=raw_dir,
        out_dir=out_dir,
        patterns=("*.pkl",),
        mode="npy",
        dt_s=dt_s,
    )
    heldout_stats = convert_split(
        split_name="heldout",
        raw_dir=raw_heldout_dir,
        out_dir=out_heldout_dir,
        patterns=("*.pkl",),
        mode="jsonl",
        dt_s=dt_s,
    )
    return set_stats, heldout_stats


def main() -> None:
    set_stats, heldout_stats = convert()
    print(
        "\nSummary (set):"
        f" seen={set_stats.files_seen},"
        f" saved={set_stats.files_converted},"
        f" reused={set_stats.files_reused},"
        f" skipped_empty={set_stats.files_skipped_empty}"
    )
    print(
        "Summary (heldout):"
        f" seen={heldout_stats.files_seen},"
        f" saved={heldout_stats.files_converted},"
        f" reused={heldout_stats.files_reused},"
        f" skipped_empty={heldout_stats.files_skipped_empty}"
    )


if __name__ == "__main__":
    main()
