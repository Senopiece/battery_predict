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

To inspect real examples of this processed format, start with [data/set/_inspect.ipynb](../data/set/_inspect.ipynb).

---

## Raw Datasets And Conversion

The repository keeps raw source datasets separately from the processed training set:

- `data/raw/batterylife/` contains the BatteryLife raw dataset staging area and [data/raw/batterylife/convert.py](data/raw/batterylife/convert.py), which converts raw pickle files into the shared `.npy` tensor format in `data/set/`.
- `data/raw/sk/` contains the SK raw dataset staging area and [data/raw/sk/convert.py](data/raw/sk/convert.py), which is the corresponding conversion entry point for that dataset.

Use the dataset-specific notebooks when you want to inspect source-data quirks before conversion:

- [data/raw/batterylife/inspect.ipynb](../data/raw/batterylife/inspect.ipynb)
- [data/raw/sk/inspect.ipynb](../data/raw/sk/inspect.ipynb)

Use [data/set/_inspect.ipynb](../data/set/_inspect.ipynb) when you want to inspect exactly what the model trains on after conversion.

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

Training uses a **battery-file-level train/validation split** to prevent leakage between cycles of the same cell across optimization and model selection.

Algorithm:
1. Sort all files by name (deterministic, hash-based names give stable ordering).
2. Shuffle with a seeded `random.Random(split_seed)`.
3. Let `N` be the total number of battery files after filtering, then compute `n_val = max(1, round(N * val_fraction))`.
4. Clamp `n_val` so at least 1 file remains for training.
5. Assign: first `n_val` → validation, remainder → train.

**Nuance:** `split_seed` is separate from the global training `seed`. This is intentional: you can change the model seed (for ensemble experiments) while keeping the same data split, or vice versa.

**Nuance:** the configured validation fraction is guaranteed only at the battery-file level. Since battery files can contain very different numbers of cycles (and therefore windows), the effective train/validation ratio by cycles or windows can slightly differ from the file-level split ratio. Use the `utilize_epoch_windows` analysis cell in [data/set/_inspect.ipynb](../data/set/_inspect.ipynb) to inspect the actual window counts per split.

### Manual Holdout Batteries

This repository does not create a random test split during training. Final evaluation is done against a manually held-out BatteryLife sodium-ion subset outside the train/validation split.

The current manually held-out files are:
- `NA-ion_4500-30_20250114232539_DefaultGroup_45_8`
- `NA-ion_270040-1-3-62`
- `NA-ion_270040-1-8-57`
- `NA-ion_270040-2-3-12`

Keep these files out of the training dataset when preparing the processed set used for model fitting.

---

## Window Sampling

Each training window is a **contiguous window of `cycle_window` consecutive cycles** from one battery file.

Window index construction:
- If a battery has fewer than `min_observed_cycles` valid cycles: **excluded entirely**.
- If a battery has between `min_observed_cycles` and `cycle_window` cycles: **one window** covering all cycles (shorter than `cycle_window` occurs only if #cycles < `cycle_window`).
- If a battery has more than `cycle_window` cycles: **sliding windows** with step 1, giving `num_cycles - cycle_window + 1` windows per file.

This means longer-lived batteries contribute exponentially more windows. It is a deliberate bias toward batteries with rich degradation trajectories.

**Nuance — epoch window usage:** Because longer batteries contribute many more windows, drawing all available windows per epoch would oversample batteries with many cycles. `utilize_epoch_windows` caps the number of windows drawn per training epoch via `torch.utils.data.RandomSampler`. If `utilize_epoch_windows > total_windows`, sampling is done with replacement. The same cap applies to validation via `utilize_val_epoch_windows`.

**Nuance — sampling randomness:** The random sampler for epoch sampling is seeded by Lightning's worker seeding tied to the global experiment seed. Different seeds between runs will draw different window subsets per epoch, but the split and the full window index are deterministic given `split_seed`.

---

## Collation

Within a batch, signal lengths can differ both across batteries and across cycles from the same battery. The collation function:

1. Pads all signals in the batch to `max_samples`.
2. Pads cycle sequences to `max_cycles` (the longest window in the batch). For the current processed dataset and default config, this is a pure no-op because all files exceed `cycle_window` and every sampled window has exactly `cycle_window` cycles. The padding path is kept for compatibility with future datasets/configs that may include shorter windows.
3. Constructs `signal_mask`, `sequence_mask`, `prediction_mask`, and `target_capacity_mask` for model masking and loss masking.

Let:
- `B` = batch size
- `C` = `max_cycles` in the batch
- `T` = `max_samples` in the batch

Mask and tensor meanings:
- `signals`: shape `(B, C, T, 2)`. Padded signal tensor (voltage/current channels).
- `signal_mask`: shape `(B, C, T)`. `True` where a timestep in a cycle is real data, `False` where it is timestep padding. This prevents attention/pooling from treating padded timesteps as observed signal.
- `sequence_mask`: shape `(B, C)`. `True` where a cycle slot in the window is real, `False` where it is cycle padding. This is the base mask for sequence-level operations.
- `prediction_mask`: shape `(B, C-1)`, computed as `sequence_mask[:, 1:] & sequence_mask[:, :-1]`. `True` for valid transition pairs `(t -> t+1)` where both cycles exist. Used for next-latent supervision.
- `target_capacity_mask`: shape `(B, C-1)`, computed as `capacity_valid[:, 1:] & prediction_mask`. `True` only when the transition is valid and the target cycle's discharge capacity is valid. Used for capacity supervision.

Reasoning:
- Separate masks let the pipeline handle two different notions of validity: signal-level validity (timesteps), sequence-level validity (cycles), and target-level validity (capacity labels).
- `prediction_mask` and `target_capacity_mask` can differ: a transition may be valid for latent prediction but excluded from capacity loss if capacity validity fails.

**Nuance:** Padding is always with zeros for signals (not NaN), and the corresponding mask positions are `False`. This is safe because the model zeros out padded positions after every attention and feed-forward operation.

---

## Capacity Normalization

Before computing the capacity loss, targets are normalized using training-split statistics:

$$
y_{norm} = \frac{Q_{ah} - \mu_{train}}{\sigma_{train}}
$$

`μ_train` and `σ_train` are computed from the training split only (no validation leakage). If `σ_train = 0`, it is replaced with `1.0`.

The model predicts in normalized space. For evaluation and logging, predictions are denormalized back to Ah:

$$
\hat{Q}_{ah} = \hat{y}_{norm} \cdot \sigma_{train} + \mu_{train}
$$

`capacity_mae_ah` in the logs is always in physical Ah units.