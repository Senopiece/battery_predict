# Battery Predict

Battery Predict trains a latent dynamics model over battery discharge cycles to forecast next-cycle capacity from variable-length voltage and current traces.

## Onboarding

If you are new to the repo, use this order:

1. Read [docs/data.md](docs/data.md) to understand the tensor format, capacity target, split logic, and window sampling.
2. Read [docs/model.md](docs/model.md) to understand the encoder, latent dynamics model, and decoder.
3. Read [docs/training.md](docs/training.md) to understand the Lightning training loop, losses, scheduler, and logging.
4. Open [notebooks/dataset/set.ipynb](notebooks/dataset/set.ipynb) to inspect the processed dataset directly.
5. If you need source-data context, inspect [notebooks/dataset/batterylife.ipynb](notebooks/dataset/batterylife.ipynb) and [notebooks/dataset/sk.ipynb](notebooks/dataset/sk.ipynb).

## Repository Scope

- `data/set/*.npy`: processed battery tensors with shape `(cycle, sample, channel)`.
- `data/raw/batterylife` and `data/raw/sk`: raw dataset staging areas plus dataset-specific conversion scripts.
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
3. Predict the next latent residual and decode next-cycle capacity.

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
data/
  raw/         raw source datasets and conversion scripts
  set/         processed project tensor format (.npy per battery)
src/battery_predict/
  data/        dataset and LightningDataModule
  models/      encoder, autoregressive transformer, decoder
  training/    config, LightningModule, callbacks, CLI
  utils/       shared helpers
tests/         targeted unit and smoke tests
docs/          design and data notes
```

## ClearML

Training initializes a native ClearML task and uses TensorBoard-compatible metric logging. Update the `clearml` section in `configs/default.yaml` for your server/project/task naming.

For a local-only run without ClearML, set `clearml.enabled: false` in your config.

## Data Conversion

Raw datasets are not consumed directly by the training code. They are first converted into the project tensor format under `data/set/`.

- `data/raw/batterylife/convert.py` converts the BatteryLife sodium-ion raw files from `data/raw/batterylife/set/naion/`.
- `data/raw/sk/convert.py` is the entry point for converting the SK raw dataset staged under `data/raw/sk/set/`.

If you are validating data quality or debugging preprocessing, start from the raw-dataset notebooks in `notebooks/dataset/` and then compare against the processed-set notebook.

## More Detail

- Data format notes: [docs/data.md](docs/data.md)
- Model summary: [docs/model.md](docs/model.md)
- Training workflow: [docs/training.md](docs/training.md)