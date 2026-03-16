from __future__ import annotations

import torch
from torch import nn

from battery_predict.models.encoder import CycleEncoder
from battery_predict.models.predictor import CapacityDecoder, LatentPredictor
from battery_predict.training.config import (
    DecoderConfig,
    EncoderConfig,
    PredictorConfig,
)


class LatentCapacityPredictor(nn.Module):
    def __init__(
        self,
        encoder_config: EncoderConfig,
        predictor_config: PredictorConfig,
        decoder_config: DecoderConfig,
    ):
        super().__init__()
        self.encoder = CycleEncoder(encoder_config)
        self.predictor = LatentPredictor(predictor_config)
        self.decoder = CapacityDecoder(encoder_config.latent_dim, decoder_config)

    def encode_cycles(
        self, signals: torch.Tensor, signal_mask: torch.Tensor
    ) -> torch.Tensor:
        batch, steps, samples, channels = signals.shape
        flat_signals = signals.view(batch * steps, samples, channels)
        flat_mask = signal_mask.view(batch * steps, samples)
        encoded = self.encoder(flat_signals, flat_mask)
        return encoded.view(batch, steps, -1)

    def forward(
        self,
        signals: torch.Tensor,
        signal_mask: torch.Tensor,
        sequence_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        latents = self.encode_cycles(signals, signal_mask)
        latents = latents * sequence_mask.unsqueeze(-1).to(latents.dtype)
        hidden, predicted_next_latent = self.predictor(latents, sequence_mask)
        capacity_mean, capacity_logvar = self.decoder(predicted_next_latent)
        return {
            "latents": latents,
            "predictor_hidden": hidden,
            "predicted_next_latent": predicted_next_latent,
            "target_next_latent": latents[:, 1:],
            "capacity_mean": capacity_mean,
            "capacity_logvar": capacity_logvar,
        }
