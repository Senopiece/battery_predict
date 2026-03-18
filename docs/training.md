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
3. **Forward pass:** encode all cycles in the window flat across the batch, then run the causal predictor, then decode.
4. **Loss:** compute the three-term deterministic objective and backpropagate through the corresponding paths.
5. **Backward + clip + step:** gradient clipping at `gradient_clip_val`, AdamW optimizer, cosine LR schedule.

---

## Three-Term Training Objective

### 1. Loss definition

$$
L_{total} = \alpha \cdot L_{direct} + \beta \cdot L_{pred\_latent} + \gamma \cdot L_{pred\_decode}
$$

### 2. Term roles

- **`L_pred_decode` (main):** optimizes the actual forecasting task (future capacity prediction from predicted latents).
- **`L_direct`:** accelerates encoder learning by supervising capacity directly from encoded latents.
- **`L_pred_latent`:** stabilizes autoregressive rollout by aligning predicted latents with the true latent trajectory.

### 3. Gradient flow

The predicted-latent target is detached:

$$
L_{pred\_latent} = \text{MSE}(\hat{z}_{t+k},\; \text{stop\_grad}(z_{t+k}))
$$

This avoids encoder/predictor collusion where the encoder could move the target latent to match a weak predictor. With detach, `L_pred_latent` updates predictor dynamics only, while `L_direct` still gives direct encoder supervision.

---

## Masks in Loss Computation

- `prediction_mask[b, t]` is `True` if both cycle `t` and cycle `t+1` are valid in sample `b`. Only these positions contribute to latent loss.
- `target_capacity_mask[b, t]` is `True` if `prediction_mask[b, t]` AND the capacity of cycle `t+1` passed the discharge threshold. These positions supervise `L_pred_decode`.
- `sequence_mask & capacity_valid` is used for `L_direct` so direct decode supervision only uses valid observed cycles.

This means a window can contribute latent supervision without contributing capacity supervision (for cycles that passed the `prediction_mask` but not the discharge threshold). This is expected and intentional.

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

## Teacher Forcing

The predictor runs in **teacher forcing mode**: ground-truth latent history is used to construct each next-latent prediction context during training.

---

## Logging

Logged metrics per split (`train`, `val`):

| Metric | Description |
|---|---|
| `{split}/loss` | total loss |
| `{split}/direct_loss` | `L_direct` |
| `{split}/pred_latent_loss` | `L_pred_latent` |
| `{split}/pred_decode_loss` | `L_pred_decode` |
| `{split}/capacity_mae_ah` | mean absolute error in Ah |

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