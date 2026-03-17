from __future__ import annotations

from datetime import datetime
from pathlib import Path
import random
import torch
import lightning as L
from lightning.pytorch.loggers import LitLogger

from battery_predict.data import BatteryDataModule
from battery_predict.training.callbacks import build_callbacks
from battery_predict.training.config import ExperimentConfig
from battery_predict.training.module import BatteryPredictorModule
from battery_predict.utils.seed import seed_everything_local


def resolve_seed(seed: int | None) -> int:
    if seed is None:
        return random.SystemRandom().randint(10000, 99999)
    return int(seed)


def make_run_dir(config: ExperimentConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(config.trainer.default_root_dir) / config.experiment_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def create_datamodule(config: ExperimentConfig) -> BatteryDataModule:
    datamodule = BatteryDataModule(config.data)
    datamodule.setup("fit")
    return datamodule


def create_module(
    config: ExperimentConfig,
    datamodule: BatteryDataModule,
) -> BatteryPredictorModule:
    return BatteryPredictorModule(
        config,
        capacity_mean_ah=datamodule.capacity_mean_ah,
        capacity_std_ah=datamodule.capacity_std_ah,
    )


def create_trainer(
    config: ExperimentConfig,
    run_dir: Path,
    *,
    enable_live_plot: bool = False,
) -> L.Trainer:
    callbacks = build_callbacks(config, run_dir, enable_live_plot=enable_live_plot)
    return L.Trainer(
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        precision=config.trainer.precision,
        max_epochs=config.trainer.max_epochs,
        gradient_clip_val=config.trainer.gradient_clip_val,
        log_every_n_steps=config.trainer.log_every_n_steps,
        deterministic=config.trainer.deterministic,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        default_root_dir=str(run_dir),
        callbacks=callbacks,
        logger=LitLogger(root_dir=run_dir, name=config.experiment_name),
    )


def fit_experiment(
    config: ExperimentConfig,
    *,
    enable_live_plot: bool = False,
    run_test: bool = True,
) -> tuple[L.Trainer, BatteryPredictorModule, BatteryDataModule, Path]:
    if config.seed is None:
        config.seed = resolve_seed(config.seed)
        print(f"[INFO] Generated random 5-digit seed: {config.seed}")

    torch.set_float32_matmul_precision("high")

    seed_everything_local(config.seed)
    L.seed_everything(config.seed, workers=True)
    run_dir = make_run_dir(config)
    config.save_yaml(run_dir / "config.yaml")
    datamodule = create_datamodule(config)
    # Print sample counts and percentages for train and val
    train_total = len(datamodule.train_dataset) if datamodule.train_dataset else 0
    val_total = len(datamodule.val_dataset) if datamodule.val_dataset else 0
    train_samples = (
        config.data.utilize_epoch_windows
        if config.data.utilize_epoch_windows is not None
        else train_total
    )
    val_samples = getattr(config.data, "utilize_val_epoch_windows", None)
    if val_samples is None:
        val_samples = val_total
    print(
        f"[INFO] Train samples: {train_samples} / {train_total} ({100.0 * train_samples / max(1, train_total):.1f}%)"
    )
    print(
        f"[INFO] Val samples: {val_samples} / {val_total} ({100.0 * val_samples / max(1, val_total):.1f}%)"
    )
    module = create_module(config, datamodule)
    trainer = create_trainer(config, run_dir, enable_live_plot=enable_live_plot)
    trainer.fit(module, datamodule=datamodule)
    if run_test:
        datamodule.setup("test")
        trainer.test(module, datamodule=datamodule, ckpt_path="best")
    return trainer, module, datamodule, run_dir
