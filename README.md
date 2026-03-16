# Battery Predict

Battery Predict trains a latent dynamics model over battery discharge cycles to forecast next-cycle capacity from variable-length voltage and current traces.

## Repository Scope

- `data/set/*.npy`: processed battery tensors with shape `(cycle, sample, channel)`.
- `src/battery_predict`: reusable data, model, loss, and training code.
- `notebooks/dataset`: dataset inspection notebooks.

## Modeling Goal

Each cycle contains a variable-length signal with two channels:

- voltage
- current

The training target is cycle discharge capacity, computed from the processed signal by integrating the magnitude of negative current over time:

$$
Q_{discharge} = \sum_t \max(-I_t, 0) \cdot \Delta t / 3600
$$

with $\Delta t = 1s$ because the converted dataset is uniformly resampled.

The model pipeline is:

1. Encode each cycle signal into a latent vector.
2. Model degradation dynamics across the sequence of cycle latents.
3. Predict the next latent residual and decode next-cycle capacity mean and variance.

## Environment Setup

This project targets Python 3.12 so CUDA-enabled PyTorch can be installed reliably on Windows.

The repository pin is stored in [.python-version](.python-version), and PyTorch is sourced from the CUDA 12.4 wheel index through [pyproject.toml](pyproject.toml).

Create or refresh the environment with `uv`:

```bash
uv sync --extra dev
```

If an older `.venv` is locked by another process on Windows, create a clean side environment and sync into it:

```bash
uv venv --python 3.12
uv sync --extra dev
```

## Training

- CLI training: `uv run battery-predict-train --config configs/default.yaml`
- Logging backend: ClearML (enabled by default in `configs/default.yaml`)

The code path has been verified with:

- `uv run pytest`
- a one-epoch real-data smoke fit with validation, test, and best-checkpoint restore

## Project Layout

```text
src/battery_predict/
  data/        dataset and LightningDataModule
  models/      encoder, autoregressive transformer, decoder
  training/    config, LightningModule, callbacks, CLI
  utils/       shared helpers
tests/         targeted unit and smoke tests
docs/          design and data notes
```

## ClearML

Training uses ClearML through the Lightning logger adapter. Update the `clearml` section in `configs/default.yaml` for your server/project/task naming.

For a local-only run without ClearML, set `clearml.enabled: false` in your config.

## More Detail

- Data format notes: [docs/data.md](docs/data.md)
- Model summary: [docs/model.md](docs/model.md)
- Training workflow: [docs/training.md](docs/training.md)