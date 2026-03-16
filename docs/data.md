# Data Notes

Processed tensors in `data/set` are produced by the raw conversion scripts and follow these rules:

- one `.npy` file per battery cell
- shape `(cycle, sample, channel)`
- channel `0`: voltage in volts
- channel `1`: current in amperes
- sampling interval `dt = 1s`
- shorter cycles are padded with trailing `NaN`

The training code must infer validity from `isfinite(voltage) & isfinite(current)` and must never treat padded samples as real signal.

Capacity supervision is derived from the processed current trace, not loaded from raw metadata. For each cycle:

$$
Q = \sum_t \max(-I_t, 0) \cdot \Delta t / 3600
$$

where only valid samples contribute and $\Delta t = 1s$.

Splits are created at the battery-file level to avoid leakage between cycles of the same cell.