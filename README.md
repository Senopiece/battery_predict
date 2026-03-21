# Battery Predict

**Build and train a latent degradation model to forecast battery discharge capacity from voltage/current signals.**

---

## Quick Start

### 1. Setup environment
```bash
uv sync --extra dev
```

### 2. Train the model
```bash
uv run train --config configs/default.yaml
```

### 3. Inspect results
Check `outputs/<experiment_name>/<timestamp>/` for checkpoints and logs.

---

## Repository Map

### 📊 Data
- **`data/set/`** — Processed battery tensors (`.npy` files, one per cell)
  - Quick inspect: [data/set/_inspect.ipynb](data/set/_inspect.ipynb)
- **`data/raw/batterylife/`** — BatteryLife raw dataset staging area
  - Browse raw data: [data/raw/batterylife/inspect.ipynb](data/raw/batterylife/inspect.ipynb)
  - Convert to processed format: `data/raw/batterylife/convert.py`
- **`data/raw/sk/`** — SK dataset staging area  
  - Browse: [data/raw/sk/inspect.ipynb](data/raw/sk/inspect.ipynb)
  - Convert: `data/raw/sk/convert.py`

### 💾 Source Code
- **`src/battery_predict/`**
  - `data/` — Dataset loading and Lightning DataModule  
  - `models/` — Encoder, aggregator, forecast head
  - `training/` — Config, LightningModule, training loop, CLI  
  - `utils/` — Shared utilities (split, capacity)

### 📖 Documentation
- **`docs/data.md`** — Tensor format, split strategy, masks, capacity targets
- **`docs/model.md`** — Architecture: encoder → aggregator → forecast head  
- **`docs/training.md`** — Training loop, loss, logging, holdout evaluation
- **`docs/inference.md`** — Local checkpoint web UI for interactive forecasting

### ⚙️ Configuration
- **`configs/default.yaml`** — Default training config (model, data, optimizer, scheduler)

---

## Learning Path

**New to the repo?** Follow this order:

1. **Understand the data format** → Read [docs/data.md](docs/data.md)
   - Tensor layout, split strategy, capacity computation
   - Then inspect with [data/set/_inspect.ipynb](data/set/_inspect.ipynb)

2. **Understand the model** → Read [docs/model.md](docs/model.md)
   - Three-stage pipeline: encoder → aggregator → forecast head
   - Architecture choices and design rationale

3. **Understand training** → Read [docs/training.md](docs/training.md)
   - MAE loss and capacity supervision
   - Logging and evaluation strategy

4. **Run your first training** → Execute:
   ```bash
   uv run train --config configs/default.yaml
   ```
   - Watch metrics in TensorBoard or ClearML
   - Find best checkpoint in `outputs/`

5. **Debug/Explore (optional)**  
   - Dataset analysis: [data/set/_inspect.ipynb](data/set/_inspect.ipynb)
   - Raw data inspection: [data/raw/batterylife/inspect.ipynb](data/raw/batterylife/inspect.ipynb)

---

## Key Concepts

### MAE Loss
$$L = \frac{1}{|\mathcal{V}|} \sum_{(b,t) \in \mathcal{V}} \lvert \hat{Q}_{b,t} - Q_{b,t} \rvert$$

- Mean absolute error over valid target capacity positions
- Targets are raw discharge capacity in Ah

### Train/Validation Split
- Random file-level split (no cycle leakage)
- Model selection: lowest `val/loss`
- Evaluation: manually held-out BatteryLife NA-ion files

### Manual Holdout Batteries
```
NA-ion_4500-30_20250114232539_DefaultGroup_45_8
NA-ion_270040-1-3-62
NA-ion_270040-1-8-57
NA-ion_270040-2-3-12
```
Keep these out of training; evaluate separately with best checkpoint.

---

## Experiment Tracking

### ClearML (via TensorBoard)
```bash
# Enable in configs/default.yaml (default: true)
clearml:
  enabled: true
  project_name: battery-predict
```

### Local Only
```bash
# Disable in configs/default.yaml
clearml:
  enabled: false
```

---

## Development

### Environment Setup (Locked)
```bash
uv venv --python 3.12
uv sync --extra dev
```

### Linting
```bash
uv run ruff check src/
uv run ruff format src/
```

### Python Version
- Target: Python 3.12 (see `.python-version`)
- PyTorch: CUDA 12.4 wheels (via custom index in `pyproject.toml`)

---

## Project Layout

```text
battery_predict/
├── data/
│   ├── raw/              raw datasets + conversion scripts
│   │   ├── batterylife/
│   │   │   ├── inspect.ipynb
│   │   │   └── convert.py
│   │   └── sk/
│   │       ├── inspect.ipynb
│   │       └── convert.py
│   └── set/              processed tensors (.npy)
│       └── _inspect.ipynb
├── src/battery_predict/
│   ├── data/             dataset module
│   ├── models/           encoder, aggregator, forecast head
│   ├── training/         config, Lightning module, CLI
│   └── utils/            split, capacity
├── configs/
│   └── default.yaml      experiment config
├── docs/
│   ├── data.md           tensor/split/capacity
│   ├── model.md          architecture
│   └── training.md       losses/logging/evaluation
└── pyproject.toml        dependencies & build
```

---

## More

- **ClearML setup**: See [docs/training.md](docs/training.md) section "ClearML"
- **Data conversion**: See [docs/data.md](docs/data.md) section "Raw Datasets"
- **Capacity calculation**: See [docs/data.md](docs/data.md) section "Capacity Computation"
- **Interactive inference UI**: See [docs/inference.md](docs/inference.md)
