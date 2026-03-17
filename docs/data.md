# Data

## Tensor Format

Each battery cell is stored as a single `.npy` file in `data/set/`:

- **Shape:** `(num_cycles, max_samples, 2)`
- **Channel 0:** voltage in volts
- **Channel 1:** current in amperes
- **Current sign convention:** negative current means discharge (current leaving the cell), positive current means charge
- **Sampling interval:** `dt = 1s` (uniform resampling applied at conversion time)
- **Padding:** cycles shorter than `max_samples` are padded with trailing `NaN` values

Validity must always be inferred from `isfinite(voltage) & isfinite(current)`. Padded positions must never be treated as real signal.

Filenames are 4-character base62 content hashes of the source data.

To inspect real examples of this processed format, start with [notebooks/dataset/set.ipynb](notebooks/dataset/set.ipynb).

---

## Raw Datasets And Conversion

The repository keeps raw source datasets separately from the processed training set:

- `data/raw/batterylife/` contains the BatteryLife raw dataset staging area and [data/raw/batterylife/convert.py](data/raw/batterylife/convert.py), which converts raw pickle files into the shared `.npy` tensor format in `data/set/`.
- `data/raw/sk/` contains the SK raw dataset staging area and [data/raw/sk/convert.py](data/raw/sk/convert.py), which is the corresponding conversion entry point for that dataset.

Use the dataset-specific notebooks when you want to inspect source-data quirks before conversion:

- [notebooks/dataset/batterylife.ipynb](notebooks/dataset/batterylife.ipynb)
- [notebooks/dataset/sk.ipynb](notebooks/dataset/sk.ipynb)

Use [notebooks/dataset/set.ipynb](notebooks/dataset/set.ipynb) when you want to inspect exactly what the model trains on after conversion.

---

## Capacity Computation

Capacity is **not** loaded from metadata. It is computed from the current trace for each cycle:

$$
Q_{discharge} = \sum_t \max(-I_t, 0) \cdot \Delta t \;/\; 3600
$$

Only valid (non-NaN) samples contribute. `dt = 1s` because the dataset is uniformly resampled at 1 Hz.

A cycle is considered to have a valid discharge capacity if `Q_discharge >= min_discharge_capacity_ah`. Cycles failing this threshold can be dropped before training via `data.drop_cycles_without_discharge`.

**Nuance:** The threshold `min_discharge_capacity_ah` defaults to `1e-6 Ah`, which is essentially zero. Its purpose is to filter out rest or charge cycles that contain no meaningful discharge, not to apply any specific quality filter. Raise this value if your dataset contains noisy near-zero-discharge cycles that should not supervise the model.

---

## Data Split

Splits are performed at the **battery-file level** to prevent leakage between cycles of the same cell across train/val/test.

Algorithm:
1. Sort all files by name (deterministic, hash-based names give stable ordering).
2. Shuffle with a seeded `random.Random(split_seed)`.
3. Compute `n_test = max(1, round(N * test_fraction))` and `n_val = max(1, round(N * val_fraction))`.
4. Clamp the sum so at least 1 file remains for training.
5. Assign: first `n_test` → test, next `n_val` → val, remainder → train.

**Nuance:** `split_seed` is separate from the global training `seed`. This is intentional: you can change the model seed (for ensemble experiments) while keeping the same data split, or vice versa.

**Nuance:** rounding can cause the actual split ratios to differ slightly from the configured fractions for small datasets. Use the notebook `epoch_samples` analysis cell to inspect the actual window counts.

---

## Window Sampling

Each training sample is a **contiguous window of `cycle_window` consecutive cycles** from one battery file.

Window index construction:
- If a battery has fewer than `min_observed_cycles` valid cycles: **excluded entirely**.
- If a battery has between `min_observed_cycles` and `cycle_window` cycles: **one window** covering all cycles (shorter than `cycle_window` occurs only if #cycles < `cycle_window`).
- If a battery has more than `cycle_window` cycles: **sliding windows** with step 1, giving `num_cycles - cycle_window + 1` windows per file.

This means longer-lived batteries contribute exponentially more windows. It is a deliberate bias toward batteries with rich degradation trajectories.

**Nuance — epoch sampling:** Because longer batteries contribute many more windows, drawing all available windows per epoch would oversample batteries with many cycles. `epoch_samples` caps the number of windows drawn per training epoch via `torch.utils.data.RandomSampler`. If `epoch_samples > total_windows`, sampling is done with replacement. The same cap applies to validation via `val_epoch_samples`.

**Nuance — sampling randomness:** The random sampler for epoch sampling is seeded by Lightning's worker seeding tied to the global experiment seed. Different seeds between runs will draw different window subsets per epoch, but the split and the full window index are deterministic given `split_seed`.

---

## Collation

Within a batch, cycles from different batteries may have different signal lengths. The collation function:

1. Pads all signals in the batch to `max_samples` (the longest signal in the batch, not the dataset-level max).
2. Pads cycle sequences to `max_cycles` (the longest window in the batch).
3. Constructs `signal_mask`, `sequence_mask`, `prediction_mask`, and `target_capacity_mask`.

**Nuance:** Padding is always with zeros for signals (not NaN), and the corresponding mask positions are `False`. This is safe because the model zeros out padded positions after every attention and feed-forward operation.

---

## Capacity Normalization

Before computing the capacity loss, targets are normalized using training-split statistics:

$$
y_{norm} = \frac{Q_{ah} - \mu_{train}}{\sigma_{train}}
$$

`μ_train` and `σ_train` are computed from the training split only (no val/test leakage). If `σ_train = 0`, it is replaced with `1.0`.

The model predicts in normalized space. For evaluation and logging, predictions are denormalized back to Ah:

$$
\hat{Q}_{ah} = \hat{y}_{norm} \cdot \sigma_{train} + \mu_{train}
$$

`capacity_mae_ah` in the logs is always in physical Ah units.