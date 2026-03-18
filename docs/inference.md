# Inference Web UI

This project includes a local checkpoint-backed web app for interactive forecasting on heldout JSONL files.

## What It Does

- Loads a trained checkpoint (`.ckpt`)
- Starts a local server and serves a web page (VS Code dark style)
- Lets you upload heldout JSONL (`{"V":[...],"A":[...]}` per line, one line per cycle)
- Plots true discharge capacity vs cycle number
- Lets you select a cycle range on the plot with the mouse
- Uses that span as context and predicts forward to a configurable right border
- Overlays predictions on the same plot for visual comparison

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

1. Upload a heldout JSONL file.
2. Confirm true curve is plotted.
3. Select cycle span with mouse (box select) or set fields manually:
   - Start cycle
   - End cycle
   - Right border
   - Context size (default 128)
4. Click Process.
5. Wait for loading overlay to finish; predictions are drawn over true values.
6. Adjust right border (can exceed true signal span) and Process again to extend forecast.

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
