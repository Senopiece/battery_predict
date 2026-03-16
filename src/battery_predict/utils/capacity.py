from __future__ import annotations

import numpy as np


def compute_valid_cycle_mask(cycle: np.ndarray) -> np.ndarray:
    return np.isfinite(cycle[:, 0]) & np.isfinite(cycle[:, 1])


def compute_discharge_capacity_ah(
    cycle: np.ndarray,
    dt_seconds: float,
    min_capacity_ah: float = 0.0,
) -> tuple[float, bool]:
    valid_mask = compute_valid_cycle_mask(cycle)
    if not np.any(valid_mask):
        return 0.0, False

    current = cycle[valid_mask, 1].astype(np.float64, copy=False)
    discharge_current = np.clip(-current, a_min=0.0, a_max=None)
    capacity_ah = float(discharge_current.sum() * dt_seconds / 3600.0)
    return capacity_ah, capacity_ah >= min_capacity_ah
