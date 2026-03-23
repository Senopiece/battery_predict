from __future__ import annotations
import torch.nn.functional as F
import numpy as np

import lightning as L
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from battery_predict.models.network import CapacityForecastModel
from battery_predict.training.config import ExperimentConfig


class BatteryPredictorModule(L.LightningModule):
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.model = CapacityForecastModel(
            config.encoder,
            config.aggregator,
            config.head,
        )
        self.save_hyperparameters(config.to_dict())

        # Buffer for batch-level losses in current epoch
        self._val_losses = []
        # Buffer for epoch-level validation losses (history)
        self._val_epoch_losses = []

    def on_validation_epoch_end(self):
        # Compute mean val loss for this epoch
        if len(self._val_losses) > 0:
            epoch_val_loss = float(np.mean(self._val_losses))
            self._val_epoch_losses.append(epoch_val_loss)
            # Apply Gaussian smoothing to epoch-level losses
            val_losses = np.array(self._val_epoch_losses)
            kernel_size = 15
            sigma = 1.0
            x = np.arange(kernel_size) - kernel_size // 2
            kernel = np.exp(-0.5 * (x / sigma) ** 2)
            kernel /= kernel.sum()
            smoothed = np.convolve(val_losses, kernel, mode="same")
            smoothed_val = float(smoothed[-1])
            self.log(
                "smoothed_val_loss",
                smoothed_val,
                prog_bar=True,
                on_epoch=True,
                on_step=False,
            )
        self._val_losses.clear()

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        num_offsets = batch["target_capacities_ah"].shape[1]
        return self.model(
            signals=batch["signals"],
            signal_mask=batch["signal_mask"],
            sequence_mask=batch["sequence_mask"],
            num_offsets=num_offsets,
        )

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        predictions = self(batch)  # (B, max_target_len_in_batch)
        target = batch["target_capacities_ah"]  # (B, max_target_len_in_batch)
        valid = batch["target_capacity_valid"]  # (B, max_target_len_in_batch)

        errors = (predictions - target).abs()
        valid_errors = errors[valid]
        loss = valid_errors.mean() if valid_errors.numel() > 0 else errors.mean()

        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=(stage != "train"),
            on_step=False,
            on_epoch=True,
        )
        self.log(f"{stage}_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log(f"{stage}/capacity_mae_ah", loss, on_step=False, on_epoch=True)
        return loss

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = self._shared_step(batch, "train")
        self.log("train/loss_epoch", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = self._shared_step(batch, "val")
        # Store batch loss for epoch mean
        self._val_losses.append(loss.detach().cpu().item())
        return loss

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay,
            betas=self.config.optimizer.betas,
        )

        if self.config.scheduler.name.lower() != "cosine":
            return optimizer

        warmup_epochs = self.config.scheduler.warmup_epochs
        total_epochs = max(self.config.trainer.max_epochs, warmup_epochs + 1)
        min_lr_ratio = self.config.scheduler.min_lr / self.config.optimizer.lr

        def schedule(epoch: int) -> float:
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            progress = (epoch - warmup_epochs) / float(
                max(1, total_epochs - warmup_epochs)
            )
            cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = LambdaLR(optimizer, lr_lambda=schedule)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
