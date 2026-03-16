# Training Workflow

## Environment

Use Python 3.12 with `uv`.

Standard setup:

```bash
uv sync --extra dev
```

If the default `.venv` is still attached to an older interpreter and cannot be replaced on Windows, use:

```bash
uv venv --python 3.12
uv sync --extra dev
```

## CLI

Train from the command line with the default config:

```bash
uv run battery-predict-train --config configs/default.yaml
```

The trainer writes outputs under `outputs/<experiment_name>/<timestamp>/` including:

- `config.yaml`
- `checkpoints/`

## ClearML Logging

ClearML is the default logging backend. Configure it in `configs/default.yaml`:

- `clearml.enabled`
- `clearml.project_name`
- `clearml.task_name`
- `clearml.tags`
- `clearml.output_uri`
- `clearml.offline_mode`

To disable ClearML for local debugging, set `clearml.enabled: false` and Lightning will fall back to TensorBoard logs under the run directory.

## Verified Checks

The current implementation has been verified with:

```bash
uv run pytest
```

and with a one-epoch real-data smoke run using a reduced model and sampled windows. That run completed fit, validation, test, and best-checkpoint restore successfully.