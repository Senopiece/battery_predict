from __future__ import annotations

import numpy as np

from battery_predict.utils.capacity import compute_discharge_capacity_ah


def test_compute_discharge_capacity_ah_uses_negative_current_only() -> None:
    cycle = np.array(
        [
            [3.2, 1.5],
            [3.1, -2.0],
            [3.0, -1.0],
            [2.9, 0.25],
        ],
        dtype=np.float32,
    )

    capacity_ah, valid = compute_discharge_capacity_ah(cycle, dt_seconds=1.0)
    assert valid
    assert np.isclose(capacity_ah, 3.0 / 3600.0)
