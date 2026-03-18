"""Decoder for capacity prediction from latent representations."""

from __future__ import annotations

import torch
from torch import nn

from battery_predict.training.config import DecoderConfig


class CapacityDecoder(nn.Module):
    """MLP decoder from latent space to capacity scalar."""

    def __init__(self, latent_dim: int, config: DecoderConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.mlp(latent).squeeze(-1)
