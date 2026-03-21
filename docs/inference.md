# Inference Web UI

This project includes a local checkpoint-backed web app for interactive forecasting on heldout JSONL files.

## What It Does

- Loads a trained checkpoint (`.ckpt`)
- Starts a local server and serves a web page
- Lets you upload heldout JSONL (`{"V":[...],"A":[...]}` per line, one line per cycle)
- Plots true discharge capacity vs cycle number
- Lets you drag boundary lines on the plot to define the input window and prediction horizon
- Runs a forecast and overlays predicted capacity on the same plot

## Command

From repository root:

```bash
uv run serve --checkpoint <path-to-checkpoint.ckpt>
```

Then open:

- http://127.0.0.1:8000

## Common Options

```bash
uv run serve \
  --checkpoint outputs/<exp>/<timestamp>/checkpoints/<best>.ckpt \
  --config configs/default.yaml \
  --host 127.0.0.1 \
  --port 8000
```

Available CLI options:

- `--checkpoint`: required path to model checkpoint
- `--config`: optional config YAML (if omitted, config is reconstructed from checkpoint hparams)
- `--host`: bind interface, default `127.0.0.1`
- `--port`: bind port, default `8000`

## UI Workflow

1. Upload a heldout JSONL file using the file picker.
2. The true capacity curve is plotted immediately.
3. Three draggable boundary lines appear on the plot:
   - **Left edge** (start cycle) — left boundary of the input window.
   - **Right edge** (end cycle) — right boundary of the input window.
   - **Dotted line** (right border) — how far into the future to predict.
4. Drag any boundary line horizontally to adjust. The shaded regions update in real time.
5. Click **Run forecast** to generate predictions.
6. Predictions are drawn as a dashed line over the true curve.
7. Adjust boundaries and run again to explore different windows and horizons. The right border can extend beyond the observed data to extrapolate.

Context size defaults to 128 cycles (capped internally if the selected input window is smaller).

## Input Format

Heldout file must be JSONL with one cycle per line:

```json
{"V":[1,1,1],"A":[1,1,2]}
{"V":[2,2],"A":[2,1]}
```

Rules:

- `V` and `A` must both be arrays.
- Effective cycle length is `min(len(V), len(A))`.
- Non-finite values are filtered out per cycle.
- Capacity is computed from current (`A`) at 1 Hz assumption.

## Notes

- The server does not modify training outputs.
- Uploaded datasets are kept in server memory for that run.
- Restart server to clear uploaded datasets.
