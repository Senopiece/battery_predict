from __future__ import annotations

import lightning as L
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from battery_predict.models import LatentCapacityPredictor
from battery_predict.training.config import ExperimentConfig
from battery_predict.training.losses import gaussian_nll, masked_mse, masked_mse_scalar


class BatteryPredictorModule(L.LightningModule):
    def __init__(
        self,
        config: ExperimentConfig,
        *,
        capacity_mean_ah: float,
        capacity_std_ah: float,
    ):
        super().__init__()
        self.config = config
        self.capacity_mean_ah = capacity_mean_ah
        self.capacity_std_ah = capacity_std_ah if capacity_std_ah > 0 else 1.0
        self.model = LatentCapacityPredictor(
            config.encoder,
            config.predictor,
            config.decoder,
        )
        self.save_hyperparameters(config.to_dict())

    def normalize_capacity(self, value_ah: torch.Tensor) -> torch.Tensor:
        return (value_ah - self.capacity_mean_ah) / self.capacity_std_ah

    def denormalize_capacity(self, value_norm: torch.Tensor) -> torch.Tensor:
        return value_norm * self.capacity_std_ah + self.capacity_mean_ah

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.model(
            signals=batch["signals"],
            signal_mask=batch["signal_mask"],
            sequence_mask=batch["sequence_mask"],
        )

    def _shared_step(
        self,
        batch: dict[str, torch.Tensor],
        stage: str,
    ) -> torch.Tensor:
        outputs = self(batch)
        prediction_mask = batch["prediction_mask"]
        target_capacity_mask = batch["target_capacity_mask"]

        latent_loss = masked_mse(
            outputs["predicted_next_latent"],
            outputs["target_next_latent"],
            prediction_mask,
        )

        target_capacity_norm = self.normalize_capacity(batch["capacities_ah"][:, 1:])
        if self.config.loss.learn_gaussian_likelihood:
            capacity_loss, clamped_logvar = gaussian_nll(
                outputs["capacity_mean"],
                outputs["capacity_logvar"],
                target_capacity_norm,
                target_capacity_mask,
                logvar_min=self.config.loss.logvar_min,
                logvar_max=self.config.loss.logvar_max,
                eps=self.config.loss.capacity_eps,
            )
        else:
            capacity_loss = masked_mse_scalar(
                outputs["capacity_mean"],
                target_capacity_norm,
                target_capacity_mask,
            )

        total_loss = capacity_loss + self.config.loss.latent_weight * latent_loss

        predicted_capacity_ah = self.denormalize_capacity(outputs["capacity_mean"])
        target_capacity_ah = batch["capacities_ah"][:, 1:]
        abs_error = (predicted_capacity_ah - target_capacity_ah).abs()
        masked_abs_error = abs_error[target_capacity_mask]
        capacity_mae = (
            masked_abs_error.mean()
            if masked_abs_error.numel() > 0
            else torch.zeros((), device=self.device)
        )

        self.log(
            f"{stage}/loss",
            total_loss,
            prog_bar=(stage != "train"),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"{stage}_loss", total_loss, prog_bar=False, on_step=False, on_epoch=True
        )
        self.log(f"{stage}/capacity_loss", capacity_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}/latent_loss", latent_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}/capacity_mae_ah", capacity_mae, on_step=False, on_epoch=True)
        if self.config.loss.learn_gaussian_likelihood:
            self.log(
                f"{stage}/logvar_mean",
                (
                    clamped_logvar[target_capacity_mask].mean()
                    if target_capacity_mask.any()
                    else torch.zeros((), device=self.device)
                ),
                on_step=False,
                on_epoch=True,
            )
        return total_loss

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = self._shared_step(batch, "train")
        self.log("train/loss_epoch", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._shared_step(batch, "val")

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
