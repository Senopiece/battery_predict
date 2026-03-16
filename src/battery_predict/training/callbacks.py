from __future__ import annotations

from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from battery_predict.training.config import CallbackConfig, ExperimentConfig


class LiveLossPlotCallback(L.Callback):
    def __init__(self):
        super().__init__()
        self.train_loss: list[float] = []
        self.val_loss: list[float] = []

    def _render(self) -> None:
        try:
            import matplotlib.pyplot as plt
            from IPython.display import clear_output, display
        except Exception:
            return

        clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        if self.train_loss:
            ax.plot(self.train_loss, label="train")
        if self.val_loss:
            ax.plot(self.val_loss, label="val")
        ax.set_title("Loss History")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        display(fig)
        plt.close(fig)

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        metric = trainer.callback_metrics.get("train/loss_epoch")
        if metric is not None:
            self.train_loss.append(float(metric.detach().cpu()))

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        metric = trainer.callback_metrics.get("val/loss")
        if metric is not None:
            self.val_loss.append(float(metric.detach().cpu()))
            self._render()


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
    return callbacks
