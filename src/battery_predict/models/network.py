from __future__ import annotations

import torch
from torch import nn

from battery_predict.models.encoder import CycleEncoder, SignalTransformerBlock
from battery_predict.models.layers import MaskedAttentionPooling
from battery_predict.training.config import AggregatorConfig, EncoderConfig, HeadConfig


def _sinusoidal_embedding(offsets: torch.Tensor, dim: int) -> torch.Tensor:
    """Fixed sinusoidal positional embedding.  offsets: (n,) long → (n, dim) float."""
    half = dim // 2
    freqs = torch.pow(
        10000.0,
        -torch.arange(half, device=offsets.device, dtype=torch.float32) / half,
    )  # (half,)
    angles = offsets.float().unsqueeze(1) * freqs.unsqueeze(0)  # (n, half)
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)  # (n, dim) when dim is even
    return emb[:, :dim]  # trim to exact dim if odd


class CapacityForecastModel(nn.Module):
    """Encode context window of cycles → pool to single latent → predict future capacities at offsets."""

    def __init__(
        self,
        encoder_config: EncoderConfig,
        aggregator_config: AggregatorConfig,
        head_config: HeadConfig,
        pred_seq_len: int,
    ):
        super().__init__()
        self.pred_seq_len = pred_seq_len
        latent_dim = encoder_config.latent_dim
        self.latent_dim = latent_dim

        # Per-cycle signal encoder (unchanged from original design)
        self.encoder = CycleEncoder(encoder_config)

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
            nn.Linear(latent_dim * aggregator_config.pooling_heads, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # Forecast head: (context_latent || offset_embed) → scalar capacity
        self.head = nn.Sequential(
            nn.Linear(latent_dim * 2, head_config.hidden_dim),
            nn.GELU(),
            nn.Linear(head_config.hidden_dim, 1),
        )

    def encode_context(
        self,
        signals: torch.Tensor,  # (B, W, T, 2)
        signal_mask: torch.Tensor,  # (B, W, T)
        sequence_mask: torch.Tensor,  # (B, W)
    ) -> torch.Tensor:  # (B, latent_dim)
        B, W, T, C = signals.shape
        flat_signals = signals.view(B * W, T, C)
        flat_mask = signal_mask.view(B * W, T)
        cycle_latents = self.encoder(flat_signals, flat_mask)  # (B*W, latent_dim)
        cycle_latents = cycle_latents.view(B, W, -1)  # (B, W, latent_dim)
        cycle_latents = cycle_latents * sequence_mask.unsqueeze(-1).to(
            cycle_latents.dtype
        )

        hidden = cycle_latents
        for block in self.cycle_transformer:
            hidden = block(hidden, sequence_mask)  # (B, W, latent_dim)
        pooled = self.pool(hidden, sequence_mask)  # (B, latent_dim * pooling_heads)
        return self.pool_proj(pooled)  # (B, latent_dim)

    def predict_at_offsets(
        self,
        context_latent: torch.Tensor,  # (B, latent_dim)
        offsets: torch.Tensor,  # (n,) long
    ) -> torch.Tensor:  # (B, n)
        B = context_latent.shape[0]
        n = offsets.shape[0]
        offset_embs = _sinusoidal_embedding(offsets, self.latent_dim)  # (n, latent_dim)
        ctx = context_latent.unsqueeze(1).expand(-1, n, -1)  # (B, n, latent_dim)
        offset_embs = offset_embs.unsqueeze(0).expand(B, -1, -1)  # (B, n, latent_dim)
        mlp_input = torch.cat([ctx, offset_embs], dim=-1)  # (B, n, latent_dim*2)
        return self.head(mlp_input).squeeze(-1)  # (B, n)

    def forward(
        self,
        signals: torch.Tensor,
        signal_mask: torch.Tensor,
        sequence_mask: torch.Tensor,
    ) -> torch.Tensor:  # (B, pred_seq_len)
        context_latent = self.encode_context(signals, signal_mask, sequence_mask)
        offsets = torch.arange(self.pred_seq_len, device=signals.device)
        return self.predict_at_offsets(context_latent, offsets)
