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
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles

from battery_predict.training.config import ExperimentConfig, load_experiment_config
from battery_predict.training.module import BatteryPredictorModule
from battery_predict.utils.capacity import compute_discharge_capacity_ah


app = FastAPI(title="Battery Predictor Local Server")
STATIC_DIR = Path(__file__).with_name("static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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

    steps_requested = right_border - end_cycle
    if steps_requested <= 0:
        return []

    pred_seq_len: int = MODEL.model.pred_seq_len
    n_to_predict = min(steps_requested, pred_seq_len)

    left = start_cycle - 1
    right = end_cycle
    context_cycles = dataset.cycles[left:right]
    if len(context_cycles) < 1:
        raise ValueError("Selected span has no cycles.")
    if len(context_cycles) > context_size:
        context_cycles = context_cycles[-context_size:]

    with torch.no_grad():
        signals, signal_mask = _build_encoder_inputs(context_cycles)
        sequence_mask = torch.ones(
            (1, len(context_cycles)), dtype=torch.bool, device=DEVICE
        )
        context_latent = MODEL.model.encode_context(signals, signal_mask, sequence_mask)
        offsets = torch.arange(n_to_predict, device=DEVICE)
        pred_caps = MODEL.model.predict_at_offsets(context_latent, offsets).squeeze(0)

    return [
        {"cycle": float(end_cycle + i + 1), "capacity_ah": float(pred_caps[i])}
        for i in range(n_to_predict)
    ]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


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
