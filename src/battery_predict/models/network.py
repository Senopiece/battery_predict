from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from battery_predict.models.embeddings import (
    RotaryPositionalEncoding,
    SinusoidalEmbedding,
)
from battery_predict.models.encoder import CycleEncoder, SignalTransformerBlock
from battery_predict.models.layers import MaskedAttentionPooling, ConstrainedLinear
from battery_predict.training.config import AggregatorConfig, EncoderConfig, HeadConfig


class CapacityForecastModel(nn.Module):
    """Encode context window of cycles → pool to single latent → predict future capacities at offsets."""

    def __init__(
        self,
        encoder_config: EncoderConfig,
        aggregator_config: AggregatorConfig,
        head_config: HeadConfig,
    ):
        super().__init__()
        latent_dim = encoder_config.latent_dim
        agg_out_dim = aggregator_config.out_dim
        offset_dim = head_config.offset_embedding_dim
        self.latent_dim = latent_dim
        self.agg_out_dim = agg_out_dim

        # Per-cycle signal encoder
        self.encoder = CycleEncoder(encoder_config)

        # Rotary positional encoding applied to cycle latent sequence
        self.cycle_rope = RotaryPositionalEncoding(
            latent_dim, base=aggregator_config.rotary_base
        )

        # Non-causal transformer over cycle latent sequence
        self.cycle_transformer = nn.ModuleList(
            [
                SignalTransformerBlock(
                    d_model=latent_dim,
                    num_heads=aggregator_config.attention_heads,
                    ff_dim=aggregator_config.ff_dim,
                    dropout=aggregator_config.dropout,
                )
                for _ in range(aggregator_config.layers)
            ]
        )

        # Attention pooling collapses (W, latent_dim) → single vector
        self.pool = MaskedAttentionPooling(latent_dim, aggregator_config.pooling_heads)
        self.pool_proj = nn.Sequential(
            nn.Linear(latent_dim * aggregator_config.pooling_heads, agg_out_dim),
            nn.LayerNorm(agg_out_dim),
        )

        # Forecast head: (context_latent || offset_embed) → scalar capacity
        self.offset_embed = SinusoidalEmbedding(offset_dim)
        self.head = nn.Sequential(
            nn.Linear(agg_out_dim + offset_dim, head_config.hidden_dim),
            nn.GELU(),
            ConstrainedLinear(head_config.hidden_dim, 1, activation=F.gelu),
        )

    def compute_last_cycle_discharge_capacity(
        self,
        signals: torch.Tensor,  # (B, W, T, 2)
        signal_mask: torch.Tensor,  # (B, W, T)
        sequence_mask: torch.Tensor,  # (B, W)
        dt_seconds: float = 1.0,
    ) -> torch.Tensor:
        """Compute discharge capacity (Ah) of the last valid context cycle per batch item."""
        B, W, T, _ = signals.shape
        valid_counts = sequence_mask.long().sum(dim=1)
        last_idx = (valid_counts - 1).clamp(min=0)

        batch_idx = torch.arange(B, device=signals.device)
        last_cycle_signals = signals[batch_idx, last_idx]  # (B, T, 2)
        last_cycle_mask = signal_mask[batch_idx, last_idx]  # (B, T)

        current = last_cycle_signals[..., 1]
        discharge_current = torch.clamp(-current, min=0.0)
        discharge_current = discharge_current * last_cycle_mask.to(
            discharge_current.dtype
        )
        return discharge_current.sum(dim=1) * (dt_seconds / 3600.0)

    def encode_context(
        self,
        signals: torch.Tensor,  # (B, W, T, 2)
        signal_mask: torch.Tensor,  # (B, W, T)
        sequence_mask: torch.Tensor,  # (B, W)
    ) -> torch.Tensor:  # (B, agg_out_dim)
        B, W, T, C = signals.shape
        flat_signals = signals.view(B * W, T, C)
        flat_mask = signal_mask.view(B * W, T)
        cycle_latents = self.encoder(flat_signals, flat_mask)  # (B*W, latent_dim)
        cycle_latents = cycle_latents.view(B, W, -1)  # (B, W, latent_dim)
        cycle_latents = cycle_latents * sequence_mask.unsqueeze(-1).to(
            cycle_latents.dtype
        )

        hidden = cycle_latents
        # Apply rotary positional encoding to inject cycle ordering
        hidden = self.cycle_rope(hidden)
        hidden = hidden * sequence_mask.unsqueeze(-1).to(hidden.dtype)
        for block in self.cycle_transformer:
            hidden = block(hidden, sequence_mask)  # (B, W, latent_dim)
        pooled = self.pool(hidden, sequence_mask)  # (B, latent_dim * pooling_heads)
        return self.pool_proj(pooled)  # (B, agg_out_dim)

    def predict_residual_at_offsets(
        self,
        context_latent: torch.Tensor,  # (B, agg_out_dim)
        offsets: torch.Tensor,  # (n,) long
    ) -> torch.Tensor:  # (B, n)
        B = context_latent.shape[0]
        n = offsets.shape[0]
        offset_embs = self.offset_embed(offsets)  # (n, offset_dim)
        ctx = context_latent.unsqueeze(1).expand(-1, n, -1)  # (B, n, agg_out_dim)
        offset_embs = offset_embs.unsqueeze(0).expand(B, -1, -1)  # (B, n, offset_dim)
        mlp_input = torch.cat(
            [ctx, offset_embs], dim=-1
        )  # (B, n, agg_out_dim+offset_dim)
        return self.head(mlp_input).squeeze(-1)  # (B, n)

    def predict_at_offsets(
        self,
        context_latent: torch.Tensor,  # (B, agg_out_dim)
        offsets: torch.Tensor,  # (n,) long
        last_cycle_capacity_ah: torch.Tensor,  # (B,)
    ) -> torch.Tensor:  # (B, n)
        residual = self.predict_residual_at_offsets(context_latent, offsets)
        return last_cycle_capacity_ah.unsqueeze(1) - torch.cumsum(residual, dim=1)

    def forward(
        self,
        signals: torch.Tensor,
        signal_mask: torch.Tensor,
        sequence_mask: torch.Tensor,
        num_offsets: int,
    ) -> torch.Tensor:  # (B, num_offsets)
        context_latent = self.encode_context(signals, signal_mask, sequence_mask)
        last_cycle_capacity_ah = self.compute_last_cycle_discharge_capacity(
            signals, signal_mask, sequence_mask
        )
        offsets = torch.arange(num_offsets, device=signals.device)
        return self.predict_at_offsets(context_latent, offsets, last_cycle_capacity_ah)
