from __future__ import annotations
from pathlib import Path
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from battery_predict.training.config import CallbackConfig, ExperimentConfig


class PrintSampleUsageCallback(L.Callback):
    def on_fit_start(self, trainer, pl_module):
        datamodule = trainer.datamodule
        if datamodule is None:
            print("[INFO] No datamodule found for sample usage printout.")
            return
        train_total = len(datamodule.train_dataset) if datamodule.train_dataset else 0
        val_total = len(datamodule.val_dataset) if datamodule.val_dataset else 0
        config = getattr(datamodule, "config", None)
        if config is not None:
            train_samples = (
                config.utilize_epoch_windows
                if getattr(config, "utilize_epoch_windows", None) is not None
                else train_total
            )
            val_samples = getattr(config, "utilize_val_epoch_windows", None)
            if val_samples is None:
                val_samples = val_total
        else:
            train_samples = train_total
            val_samples = val_total
        print(
            f"[INFO] Train samples: {train_samples} / {train_total} ({100.0 * train_samples / max(1, train_total):.1f}%)"
        )
        print(
            f"[INFO] Val samples: {val_samples} / {val_total} ({100.0 * val_samples / max(1, val_total):.1f}%)"
        )


def build_callbacks(
    config: ExperimentConfig,
    run_dir: Path,
    *,
    enable_live_plot: bool = False,
) -> list[L.Callback]:
    callback_cfg: CallbackConfig = config.callbacks
    callbacks: list[L.Callback] = [
        ModelCheckpoint(
            dirpath=run_dir / "checkpoints",
            monitor=callback_cfg.checkpoint_monitor,
            mode=callback_cfg.checkpoint_mode,
            filename=callback_cfg.checkpoint_filename,
            auto_insert_metric_name=False,
            save_top_k=callback_cfg.save_top_k,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if enable_live_plot:
        callbacks.append(LiveLossPlotCallback())
    callbacks.append(PrintSampleUsageCallback())
    return callbacks
