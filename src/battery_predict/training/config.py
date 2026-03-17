from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DataConfig:
    dataset_dir: str = "data/set"
    split_seed: int = 7
    val_fraction: float = 0.2
    test_fraction: float = 0.1
    cycle_window: int = 12
    min_observed_cycles: int = 2
    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    utilize_epoch_windows: int | None = 1024
    utilize_val_epoch_windows: int | None = 1024
    dt_seconds: float = 1.0
    min_discharge_capacity_ah: float = 1e-6
    drop_cycles_without_discharge: bool = True


@dataclass(slots=True)
class EncoderConfig:
    d_model: int = 64
    latent_dim: int = 64
    conv_channels: tuple[int, ...] = (32, 64)
    conv_kernels: tuple[int, ...] = (5, 3)
    conv_strides: tuple[int, ...] = (1, 2)
    conv_group_norm_groups: int = 8
    transformer_layers: int = 1
    attention_heads: int = 4
    ff_dim: int = 128
    dropout: float = 0.1
    pooling_heads: int = 2
    max_signal_positions: int = 2048


@dataclass(slots=True)
class PredictorConfig:
    d_model: int = 64
    layers: int = 2
    attention_heads: int = 4
    ff_dim: int = 128
    dropout: float = 0.1
    max_cycle_positions: int = 64
    rotary_base: float = 10000.0


@dataclass(slots=True)
class DecoderConfig:
    hidden_dim: int = 64


@dataclass(slots=True)
class ScheduledSamplingConfig:
    enabled: bool = False
    schedule: str = "linear"
    start_probability: float = 0.0
    end_probability: float = 0.0
    warmup_epochs: int = 0
    max_unroll_steps: int = 0


@dataclass(slots=True)
class LossConfig:
    latent_weight: float = 0.5
    learn_gaussian_likelihood: bool = True
    logvar_min: float = -10.0
    logvar_max: float = 3.0
    capacity_eps: float = 1e-6


@dataclass(slots=True)
class OptimizerConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.95)


@dataclass(slots=True)
class SchedulerConfig:
    name: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-5


@dataclass(slots=True)
class TrainerConfig:
    max_epochs: int = 30
    accelerator: str = "auto"
    devices: int | str = "auto"
    precision: str = "16-mixed"
    gradient_clip_val: float = 1.0
    log_every_n_steps: int = 10
    deterministic: bool = False
    accumulate_grad_batches: int = 1
    default_root_dir: str = "outputs"


@dataclass(slots=True)
class CallbackConfig:
    checkpoint_monitor: str = "val/loss"
    checkpoint_mode: str = "min"
    checkpoint_filename: str = "best-{epoch:02d}-{val_loss:.4f}"
    save_top_k: int = 1


@dataclass(slots=True)
class ClearMLConfig:
    enabled: bool = True
    project_name: str = "battery-predict"
    task_name: str = "latent_capacity_predictor"
    tags: tuple[str, ...] = ("battery", "lightning", "transformer")
    output_uri: str | None = None
    offline_mode: bool = False


@dataclass(slots=True)
class ExperimentConfig:
    experiment_name: str = "latent_capacity_predictor"
    seed: int | None = None
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    scheduled_sampling: ScheduledSamplingConfig = field(
        default_factory=ScheduledSamplingConfig
    )
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    callbacks: CallbackConfig = field(default_factory=CallbackConfig)
    clearml: ClearMLConfig = field(default_factory=ClearMLConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_yaml(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_experiment_config(path: str | Path | None = None) -> ExperimentConfig:
    config = ExperimentConfig()
    if path is None:
        return config

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise TypeError("Config file must contain a mapping at the top level.")

    return _merge_dataclass(config, payload)
