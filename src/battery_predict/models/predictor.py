"""Latent sequence predictor for capacity prediction chains."""

from __future__ import annotations

import torch
from torch import nn

from battery_predict.models.attention import PredictorBlock
from battery_predict.training.config import PredictorConfig


class LatentPredictor(nn.Module):
    """Autoregressive predictor over latent capacity sequences."""

    def __init__(self, config: PredictorConfig):
        super().__init__()
        self.blocks = nn.ModuleList(
            [PredictorBlock(config) for _ in range(config.layers)]
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.residual_head = nn.Linear(config.d_model, config.d_model)

    def forward(
        self, latents: torch.Tensor, sequence_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = latents
        for block in self.blocks:
            hidden = block(hidden, sequence_mask)
        hidden = self.norm(hidden) * sequence_mask.unsqueeze(-1).to(latents.dtype)
        residual = self.residual_head(hidden[:, :-1])
        next_latent = latents[:, :-1] + residual
        return hidden, next_latent
