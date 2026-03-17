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
uv run battery-predict-train --config configs/default.yaml
```

Optional flags:
- `--skip-test` — skip the test-set evaluation after fitting.

Outputs are written under `outputs/<experiment_name>/<timestamp>/`:
- `config.yaml` — the resolved config with the actual seed (if `seed: null` was used).
- `checkpoints/` — Lightning checkpoints, best-model selected by `val/loss`.

---

## Training Loop

Training is implemented with PyTorch Lightning (`BatteryPredictorModule`). The loop is:

1. **Setup:** datamodule loads all files, builds per-split window indices, fits capacity normalization stats on the training split.
2. **Epoch:** draw up to `utilize_epoch_windows` windows from train and `utilize_val_epoch_windows` from val.
3. **Forward pass:** encode all cycles in the window flat across the batch, then run the causal predictor, then decode.
4. **Loss:** compute latent loss and capacity loss, sum with `latent_weight`.
5. **Backward + clip + step:** gradient clipping at `gradient_clip_val`, AdamW optimizer, cosine LR schedule.

After training, the test split is evaluated using the best checkpoint (lowest `val/loss`).

---

## Loss Functions

### Latent loss

Masked MSE between predicted next-cycle latent and the target (actual next-cycle latent from the encoder):

$$
L_{latent} = \frac{\sum_{b,t} m_{b,t} \cdot \|\hat{z}_{b,t} - z_{b,t+1}\|^2 / D}{\max\!\left(\sum_{b,t} m_{b,t},\, 1\right)}
$$

where $m$ is `prediction_mask`, $D$ is `latent_dim`, and the mean is over the latent dimension.

### Capacity loss

Two modes controlled by `loss.learn_gaussian_likelihood`:

**Deterministic (default, `false`):** masked MSE on normalized capacity mean:

$$
L_{cap} = \frac{\sum_{b,t} m^{cap}_{b,t} \cdot (\hat{y}_{b,t} - y_{b,t})^2}{\max\!\left(\sum_{b,t} m^{cap}_{b,t},\, 1\right)}
$$

**Gaussian NLL (`true`):** masked negative log-likelihood of a Gaussian with predicted mean and log-variance:

$$
L_{cap} = \frac{\sum_{b,t} m^{cap}_{b,t} \cdot \left[\frac{(\hat{y} - y)^2}{e^{\hat{\sigma}^2}} + \hat{\sigma}^2 + \log(2\pi)\right] / 2}{\max\!\left(\sum m^{cap},\, 1\right)}
$$

where $\hat{\sigma}^2$ is the predicted log-variance clamped to `[logvar_min, logvar_max]`.

### Total loss

$$
L = L_{cap} + \lambda_{latent} \cdot L_{latent}
$$

with `loss.latent_weight = λ_latent`.

**Nuance — latent loss is in normalized latent space:** there is no explicit normalization of the latent vectors, so `latent_weight` may need tuning if the latent magnitudes grow or shrink significantly during training. Watch `latent_loss` vs `capacity_loss` in the logs to diagnose imbalance.

**Nuance — `capacity_eps`:** the Gaussian variance is clamped from below with `capacity_eps` before dividing. This prevents numerical instability when the predicted variance is very close to zero. Only relevant when `learn_gaussian_likelihood: true`.

---

## Masks in Loss Computation

- `prediction_mask[b, t]` is `True` if both cycle `t` and cycle `t+1` are valid in sample `b`. Only these positions contribute to latent loss.
- `target_capacity_mask[b, t]` is `True` if `prediction_mask[b, t]` AND the capacity of cycle `t+1` passed the discharge threshold. Only these positions contribute to capacity loss.

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

`seed: null` in config generates a random 5-digit integer seed at runtime, prints it, and saves it to the run's `config.yaml`. This allows full reproducibility after the fact even for exploratory runs.

**Nuance:** `data.split_seed` is independent of the global seed and controls only the battery-file split. This separation allows sweeping model hyperparameters or seeds without changing the train/val/test file assignment.

---

## Teacher Forcing

The predictor currently runs in **teacher forcing mode** only: the ground-truth latent from the encoder is always passed as input to the predictor at every step. There is no rollout or scheduled sampling active in the current version, even though the `scheduled_sampling` config section exists. See `scheduled_sampling.enabled` — it is always `false` and the config is not wired to any training code.

---

## Logging

Logged metrics per split (`train`, `val`, `test`):

| Metric | Description |
|---|---|
| `{split}/loss` | total loss |
| `{split}/capacity_loss` | capacity component of loss |
| `{split}/latent_loss` | latent MSE component |
| `{split}/capacity_mae_ah` | mean absolute error in Ah (denormalized) |
| `{split}/logvar_mean` | mean predicted log-variance *(only when `learn_gaussian_likelihood: true`)* |

Metrics are logged via LitLogger to the [Lightning Experiments](https://lightning.ai/) platform. Local log files are written under `<run_dir>/` (i.e. `outputs/<experiment_name>/<timestamp>/`). When logged in to Lightning AI, runs also appear in the cloud dashboard automatically.

---

## Checkpointing

Lightning `ModelCheckpoint` monitors `val/loss`, saves the top-1 best checkpoint. Checkpoint filename pattern: `best-{epoch:02d}-{val_loss:.4f}.ckpt`.

After fit, the test pass loads the best checkpoint via `ckpt_path="best"`.

---

## Running Tests

```bash
uv run pytest
```