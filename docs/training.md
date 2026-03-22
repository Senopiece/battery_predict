# Training

## Environment Setup

Use Python 3.12 with `uv`:

```bash
uv sync --extra dev
```

If the default `.venv` is locked by another process on Windows:

```bash
uv venv --python 3.12
uv sync --extra dev
```

---

## Running Training

```bash
uv run train --config configs/default.yaml
```

Outputs are written under `outputs/<experiment_name>/<timestamp>/`:
- `config.yaml` — the resolved config with the actual seed (if `seed: null` was used).
- `checkpoints/` — Lightning checkpoints, best-model selected by `val/loss`.

---

## Training Loop

Training is implemented with PyTorch Lightning (`BatteryPredictorModule`). The loop is:

1. **Setup:** datamodule loads all files and builds train/validation window indices.
2. **Epoch:** draw up to `utilize_epoch_windows` windows from train and `utilize_val_epoch_windows` from val.
3. **Forward pass:** encode the fixed-length context window, aggregate cycle latents into a context vector, then predict the full remaining future trajectory for each sample. Because different samples can have different remaining horizons, targets are padded to the batch maximum length and masked.
4. **Loss:** compute masked MAE between predicted and target capacities; backpropagate.
5. **Backward + clip + step:** gradient clipping at `gradient_clip_val`, AdamW optimizer, cosine LR schedule.

---

## Loss

### Definition

The model uses a single **mean absolute error (MAE)** loss over predicted future capacities:

$$
L = \frac{1}{|\mathcal{V}|} \sum_{(b,t) \in \mathcal{V}} \lvert \hat{Q}_{b,t} - Q_{b,t} \rvert
$$

where $\mathcal{V}$ is the set of positions where `target_capacity_valid` is `True`.

### Masking

- `target_capacity_valid[b, t]` is `True` if the target cycle at offset `t` exists in the battery and its discharge capacity passed the validity threshold.
- Different samples in the same batch can have different target horizons; right-padded target positions are marked invalid and do not contribute to the loss.
- If no valid targets exist in a batch (edge case), the loss falls back to the full unmasked error mean.

---

## Optimizer and Scheduler

**AdamW** with:
- `lr`, `weight_decay`, `betas` from config.
- Gradient clipping by global norm at `gradient_clip_val`.

**Cosine LR schedule** with linear warmup:
- Warm up linearly from 0 to `lr` over `scheduler.warmup_epochs`.
- Cosine decay from `lr` to `scheduler.min_lr` over the remaining epochs.
- If `warmup_epochs >= max_epochs`, the schedule is effectively linear and never reaches the cosine phase.

**Nuance — `accumulate_grad_batches`:** setting this > 1 accumulates gradients over multiple batches before each optimizer step. Effectively multiplies the batch size without extra memory, but does increase per-step compute.

---

## Seeding

The global seed controls:
- Python `random`, `numpy`, `torch` (including CUDA seeds).
- Lightning worker seeds.

`seed: null` in config generates a random 5-digit integer seed at runtime, prints it, and saves it to the run's `config.yaml` and ClearML config payload. This allows full reproducibility after the fact even for exploratory runs.

**Nuance:** `data.split_seed` is independent of the global seed and controls only the battery-file train/validation split. This separation allows sweeping model hyperparameters or seeds without changing the file assignment.

---

## Logging

Logged metrics per split (`train`, `val`):

| Metric | Description |
|---|---|
| `{split}/loss` | MAE loss |
| `{split}_loss` | same, flat key for checkpoint callbacks |
| `{split}/capacity_mae_ah` | mean absolute error in Ah (equal to loss) |

**ClearML:** when `clearml.enabled: true`, a ClearML Task is initialized via `Task.init(...)` before training. The resolved config dict is attached to the task as a "config" parameter group. Metrics are reported via TensorBoard logger and automatically picked up by the ClearML agent. Offline mode writes to a local task package instead of the server.

**Fallback:** when ClearML is disabled, a TensorBoard logger writes event files under `<run_dir>/tb/`.

---

## Checkpointing

Lightning `ModelCheckpoint` monitors `val/loss`, saves the top-1 best checkpoint. Checkpoint filename pattern: `best-{epoch:02d}-{val_loss:.4f}.ckpt`.

---

## Manual Holdout Evaluation

Training does not run a random test split or an automatic post-fit test pass.

Model selection is based on the validation split only. Final evaluation is reserved for manually held-out BatteryLife sodium-ion files:
- `NA-ion_4500-30_20250114232539_DefaultGroup_45_8`
- `NA-ion_270040-1-3-62`
- `NA-ion_270040-1-8-57`
- `NA-ion_270040-2-3-12`

Those holdouts should remain outside the processed training set and be evaluated manually with the checkpoint selected by `val/loss`.
