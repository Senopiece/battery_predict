from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import typer
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from battery_predict.training.config import ExperimentConfig, load_experiment_config
from battery_predict.training.module import BatteryPredictorModule
from battery_predict.utils.capacity import compute_discharge_capacity_ah


app = FastAPI(title="Battery Predictor Local Server")


class PredictRequest(BaseModel):
    dataset_id: str
    start_cycle: int = Field(ge=1)
    end_cycle: int = Field(ge=1)
    right_border: int = Field(ge=1)
    context_size: int = Field(default=128, ge=2)


@dataclass
class UploadedDataset:
    cycles: list[np.ndarray]
    capacities_ah: np.ndarray


MODEL: BatteryPredictorModule | None = None
UPLOADED: dict[str, UploadedDataset] = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _merge_config(config: ExperimentConfig, values: dict[str, Any]) -> ExperimentConfig:
    for key, value in values.items():
        if not hasattr(config, key):
            continue
        current = getattr(config, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_config(current, value)
        else:
            setattr(config, key, value)
    return config


def load_model(
    *,
    checkpoint_path: Path,
    config_path: Path | None,
) -> BatteryPredictorModule:
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")

    if config_path is not None:
        config = load_experiment_config(config_path)
    else:
        config = ExperimentConfig()
        hparams = checkpoint_payload.get("hyper_parameters", {})
        if isinstance(hparams, dict):
            _merge_config(config, hparams)

    module = BatteryPredictorModule.load_from_checkpoint(
        str(checkpoint_path),
        map_location="cpu",
        config=config,
        strict=False,
    )
    module.eval()
    module.to(DEVICE)
    return module


def parse_jsonl_cycles(file_bytes: bytes) -> list[np.ndarray]:
    lines = file_bytes.decode("utf-8").splitlines()
    cycles: list[np.ndarray] = []

    for line_idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue

        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at line {line_idx}: {exc}") from exc

        if not isinstance(payload, dict):
            raise ValueError(f"Line {line_idx}: expected object with V/A arrays.")

        voltage = payload.get("V")
        current = payload.get("A")

        if not isinstance(voltage, list) or not isinstance(current, list):
            raise ValueError(f"Line {line_idx}: expected V and A as arrays.")

        n = min(len(voltage), len(current))
        if n <= 0:
            continue

        v = np.asarray(voltage[:n], dtype=np.float32)
        a = np.asarray(current[:n], dtype=np.float32)
        finite = np.isfinite(v) & np.isfinite(a)
        if finite.sum() == 0:
            continue

        cycle = np.stack([v[finite], a[finite]], axis=-1).astype(np.float32, copy=False)
        if cycle.shape[0] > 0:
            cycles.append(cycle)

    if not cycles:
        raise ValueError("No valid cycles found in uploaded JSONL.")

    return cycles


def compute_capacities(cycles: list[np.ndarray], dt_seconds: float = 1.0) -> np.ndarray:
    capacities: list[float] = []
    for cycle in cycles:
        capacity_ah, _ = compute_discharge_capacity_ah(
            cycle,
            dt_seconds=dt_seconds,
            min_capacity_ah=0.0,
        )
        capacities.append(capacity_ah)
    return np.asarray(capacities, dtype=np.float32)


def _build_encoder_inputs(
    cycles: list[np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
    steps = len(cycles)
    max_samples = max(cycle.shape[0] for cycle in cycles)

    signals = torch.zeros(
        (1, steps, max_samples, 2), dtype=torch.float32, device=DEVICE
    )
    signal_mask = torch.zeros((1, steps, max_samples), dtype=torch.bool, device=DEVICE)

    for idx, cycle in enumerate(cycles):
        length = cycle.shape[0]
        signals[0, idx, :length, :] = torch.from_numpy(cycle).to(DEVICE)
        signal_mask[0, idx, :length] = True

    return signals, signal_mask


def run_prediction(
    *,
    dataset: UploadedDataset,
    start_cycle: int,
    end_cycle: int,
    right_border: int,
    context_size: int,
) -> list[dict[str, float]]:
    if MODEL is None:
        raise RuntimeError("Model is not loaded.")

    num_cycles = len(dataset.cycles)
    if start_cycle > end_cycle:
        raise ValueError("start_cycle must be <= end_cycle.")
    if end_cycle > num_cycles:
        raise ValueError("end_cycle exceeds available true cycles.")
    if right_border <= end_cycle:
        return []

    left = start_cycle - 1
    right = end_cycle
    context_cycles = dataset.cycles[left:right]
    if len(context_cycles) < 2:
        raise ValueError("Selected span must include at least 2 cycles.")

    if len(context_cycles) > context_size:
        context_cycles = context_cycles[-context_size:]

    with torch.no_grad():
        signals, signal_mask = _build_encoder_inputs(context_cycles)
        latents = MODEL.model.encode_cycles(signals, signal_mask)

        preds: list[dict[str, float]] = []
        steps_to_predict = right_border - end_cycle

        for step_idx in range(steps_to_predict):
            if latents.size(1) < 2:
                raise ValueError("Need at least 2 context cycles for prediction.")

            if latents.size(1) > context_size:
                latents = latents[:, -context_size:, :]

            sequence_mask = torch.ones(
                (1, latents.size(1)), dtype=torch.bool, device=DEVICE
            )
            _, predicted_next_latent = MODEL.model.predictor(latents, sequence_mask)
            next_latent = predicted_next_latent[:, -1:, :]
            pred_ah = MODEL.model.decoder(next_latent).squeeze(0).squeeze(0).item()

            pred_cycle = float(end_cycle + step_idx + 1)
            preds.append({"cycle": pred_cycle, "capacity_ah": float(pred_ah)})
            latents = torch.cat([latents, next_latent], dim=1)

    return preds


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML_PAGE


@app.post("/api/upload")
async def upload_jsonl(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename.lower().endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="Upload a .jsonl file.")

    payload = await file.read()
    try:
        cycles = parse_jsonl_cycles(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    capacities = compute_capacities(cycles)
    dataset_id = uuid.uuid4().hex
    UPLOADED[dataset_id] = UploadedDataset(cycles=cycles, capacities_ah=capacities)

    return {
        "dataset_id": dataset_id,
        "cycle_count": len(cycles),
        "true_capacities": [float(v) for v in capacities.tolist()],
    }


@app.post("/api/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    dataset = UPLOADED.get(req.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload again.")

    try:
        preds = run_prediction(
            dataset=dataset,
            start_cycle=req.start_cycle,
            end_cycle=req.end_cycle,
            right_border=req.right_border,
            context_size=req.context_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"predictions": preds}


HTML_PAGE = """
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>Battery Predictor</title>
  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
  <style>
    :root {
      --bg: #1e1e1e;
      --panel: #252526;
      --panel-2: #2d2d30;
      --text: #d4d4d4;
      --muted: #9da1a6;
      --accent: #0e639c;
      --accent-2: #3a96dd;
      --border: #3c3c3c;
      --danger: #f14c4c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Segoe UI, -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .wrap { max-width: 1200px; margin: 0 auto; padding: 16px; }
    h1 { margin: 0 0 12px; font-size: 20px; font-weight: 600; }
    .card {
      background: linear-gradient(180deg, var(--panel), var(--panel-2));
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
      margin-bottom: 12px;
    }
    .row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
    label { font-size: 12px; color: var(--muted); }
    input[type=number], input[type=file] {
      background: #1f1f1f;
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 8px;
    }
    button {
      background: var(--accent);
      color: white;
      border: 0;
      border-radius: 6px;
      padding: 9px 14px;
      cursor: pointer;
      font-weight: 600;
    }
    button:hover { background: var(--accent-2); }
    #status { font-size: 13px; color: var(--muted); }
    #plot { width: 100%; height: 62vh; min-height: 420px; }
    .num { width: 110px; }

    .overlay {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.55);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 999;
      backdrop-filter: blur(1px);
    }
    .overlay.show { display: flex; }
    .loader {
      width: min(520px, 88vw);
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 18px;
    }
    .bar {
      margin-top: 10px;
      height: 10px;
      border-radius: 99px;
      overflow: hidden;
      border: 1px solid var(--border);
      background: #111;
    }
    .bar > div {
      width: 35%;
      height: 100%;
      background: linear-gradient(90deg, transparent, var(--accent-2), transparent);
      animation: sweep 1.2s linear infinite;
      transform: translateX(-120%);
    }
    @keyframes sweep {
      from { transform: translateX(-120%); }
      to { transform: translateX(320%); }
    }
    .err { color: var(--danger); }
  </style>
</head>
<body>
  <div class=\"overlay\" id=\"overlay\">
    <div class=\"loader\">
      <div>Running model inference...</div>
      <div class=\"bar\"><div></div></div>
    </div>
  </div>

  <div class=\"wrap\">
    <h1>Local Checkpoint Inference</h1>

    <div class=\"card\">
      <div class=\"row\">
        <div>
          <label>Upload heldout JSONL</label><br/>
          <input id=\"jsonlFile\" type=\"file\" accept=\".jsonl\" />
        </div>
        <button id=\"uploadBtn\">Upload</button>
        <span id=\"status\">No dataset uploaded.</span>
      </div>
    </div>

    <div class=\"card\">
      <div class=\"row\">
        <div>
          <label>Start cycle (1-based)</label><br/>
          <input id=\"startCycle\" class=\"num\" type=\"number\" min=\"1\" value=\"1\" />
        </div>
        <div>
          <label>End cycle (1-based)</label><br/>
          <input id=\"endCycle\" class=\"num\" type=\"number\" min=\"1\" value=\"16\" />
        </div>
        <div>
          <label>Right border</label><br/>
          <input id=\"rightBorder\" class=\"num\" type=\"number\" min=\"1\" value=\"32\" />
        </div>
        <div>
          <label>Context size</label><br/>
          <input id=\"contextSize\" class=\"num\" type=\"number\" min=\"2\" value=\"128\" />
        </div>
        <button id=\"processBtn\">Process</button>
      </div>
      <div style=\"margin-top:8px;color:var(--muted);font-size:12px;\">
        Use box-select on the plot to choose a cycle range. You can extend right border beyond true span.
      </div>
    </div>

    <div class=\"card\">
      <div id=\"plot\"></div>
    </div>
  </div>

  <script>
    let datasetId = null;
    let trueCaps = [];

    const overlay = document.getElementById('overlay');
    const statusEl = document.getElementById('status');

    function setBusy(v) {
      overlay.classList.toggle('show', v);
      document.body.style.pointerEvents = v ? 'none' : '';
      overlay.style.pointerEvents = 'all';
    }

    function setStatus(msg, isError=false) {
      statusEl.textContent = msg;
      statusEl.className = isError ? 'err' : '';
    }

    function buildTrueTrace() {
      return {
        x: trueCaps.map((_, i) => i + 1),
        y: trueCaps,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'true discharge_capacity',
        line: {color: '#4fc1ff', width: 2},
        marker: {size: 4}
      };
    }

    function renderPlot(predictions=[]) {
      const traces = [buildTrueTrace()];
      if (predictions.length > 0) {
        traces.push({
          x: predictions.map(p => p.cycle),
          y: predictions.map(p => p.capacity_ah),
          type: 'scatter',
          mode: 'lines+markers',
          name: 'predicted',
          line: {color: '#ffb454', width: 2, dash: 'dash'},
          marker: {size: 4}
        });
      }

      const layout = {
        template: 'plotly_dark',
        paper_bgcolor: '#252526',
        plot_bgcolor: '#1e1e1e',
        font: {color: '#d4d4d4'},
        margin: {t: 20, r: 20, b: 50, l: 60},
        dragmode: 'select',
        xaxis: {title: 'Cycle #', zeroline: false, gridcolor: '#3c3c3c'},
        yaxis: {title: 'Discharge capacity (Ah)', zeroline: false, gridcolor: '#3c3c3c'}
      };

      Plotly.newPlot('plot', traces, layout, {responsive: true, displaylogo: false});

      const plot = document.getElementById('plot');
      plot.on('plotly_selected', (ev) => {
        if (!ev || !ev.points || ev.points.length === 0) return;
        const xs = ev.points.map(p => p.x).filter(x => Number.isFinite(x));
        if (xs.length === 0) return;
        const minX = Math.max(1, Math.floor(Math.min(...xs)));
        const maxX = Math.max(minX, Math.ceil(Math.max(...xs)));
        document.getElementById('startCycle').value = minX;
        document.getElementById('endCycle').value = maxX;
        const right = Number(document.getElementById('rightBorder').value || 0);
        if (right < maxX) {
          document.getElementById('rightBorder').value = maxX;
        }
      });
    }

    async function uploadJsonl() {
      const input = document.getElementById('jsonlFile');
      if (!input.files || input.files.length === 0) {
        setStatus('Choose a JSONL file first.', true);
        return;
      }

      setBusy(true);
      try {
        const form = new FormData();
        form.append('file', input.files[0]);

        const res = await fetch('/api/upload', {method: 'POST', body: form});
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Upload failed');

        datasetId = data.dataset_id;
        trueCaps = data.true_capacities;

        document.getElementById('startCycle').value = 1;
        document.getElementById('endCycle').value = Math.min(16, data.cycle_count);
        document.getElementById('rightBorder').value = Math.min(32, data.cycle_count);

        renderPlot([]);
        setStatus(`Uploaded ${data.cycle_count} cycles. Select span and click Process.`);
      } catch (err) {
        setStatus(err.message || String(err), true);
      } finally {
        setBusy(false);
      }
    }

    async function runProcess() {
      if (!datasetId) {
        setStatus('Upload JSONL first.', true);
        return;
      }

      const payload = {
        dataset_id: datasetId,
        start_cycle: Number(document.getElementById('startCycle').value),
        end_cycle: Number(document.getElementById('endCycle').value),
        right_border: Number(document.getElementById('rightBorder').value),
        context_size: Number(document.getElementById('contextSize').value)
      };

      setBusy(true);
      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Prediction failed');

        renderPlot(data.predictions || []);
        setStatus(`Predicted ${data.predictions.length} cycles.`);
      } catch (err) {
        setStatus(err.message || String(err), true);
      } finally {
        setBusy(false);
      }
    }

    document.getElementById('uploadBtn').addEventListener('click', uploadJsonl);
    document.getElementById('processBtn').addEventListener('click', runProcess);

    renderPlot([]);
  </script>
</body>
</html>
"""


def main(
    checkpoint: Path = typer.Option(..., help="Path to .ckpt checkpoint file."),
    config: Path | None = typer.Option(None, help="Optional config YAML path."),
    host: str = typer.Option("127.0.0.1", help="Host interface to bind."),
    port: int = typer.Option(8000, help="Port to serve the web UI."),
) -> None:
    global MODEL

    if not checkpoint.exists():
        raise typer.BadParameter(f"Checkpoint not found: {checkpoint}")

    MODEL = load_model(
        checkpoint_path=checkpoint,
        config_path=config,
    )

    print(f"[INFO] Model loaded from {checkpoint}")
    print(f"[INFO] Serving UI at http://{host}:{port}")

    uvicorn.run(app, host=host, port=port, log_level="info")


def cli() -> None:
    typer.run(main)


if __name__ == "__main__":
    cli()
