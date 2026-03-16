from __future__ import annotations

import math

import torch
from torch import nn

from battery_predict.models.layers import FeedForward
from battery_predict.training.config import DecoderConfig, PredictorConfig


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1)
    return rotated.flatten(start_dim=-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(
                "Rotary embedding requires an even attention head dimension."
            )
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.outer(positions.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def apply(self, tensor: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cos, sin = self.forward(positions)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        return (tensor * cos) + (rotate_half(tensor) * sin)


class CausalSelfAttention(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, dropout: float, rotary_base: float
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rotary = RotaryEmbedding(self.head_dim, base=rotary_base)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, steps, dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, steps, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, steps, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, steps, self.num_heads, self.head_dim).transpose(1, 2)

        positions = torch.arange(steps, device=x.device)
        q = self.rotary.apply(q, positions)
        k = self.rotary.apply(k, positions)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal_mask = torch.triu(
            torch.ones((steps, steps), device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        key_mask = ~mask.unsqueeze(1).unsqueeze(2)
        attn_scores = attn_scores.masked_fill(key_mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = torch.where(
            torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights
        )
        attn_weights = self.dropout(attn_weights)

        attended = torch.matmul(attn_weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch, steps, dim)
        attended = self.out(attended)
        return attended * mask.unsqueeze(-1).to(attended.dtype)


class PredictorBlock(nn.Module):
    def __init__(self, config: PredictorConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(
            d_model=config.d_model,
            num_heads=config.attention_heads,
            dropout=config.dropout,
            rotary_base=config.rotary_base,
        )
        self.dropout1 = nn.Dropout(config.dropout)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.ff_dim, config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout1(self.attn(self.norm1(x), mask))
        x = x * mask.unsqueeze(-1).to(x.dtype)
        x = x + self.dropout2(self.ff(self.norm2(x)))
        x = x * mask.unsqueeze(-1).to(x.dtype)
        return x


class LatentPredictor(nn.Module):
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


class CapacityDecoder(nn.Module):
    def __init__(self, latent_dim: int, config: DecoderConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 2),  # Always output 2: mean, logvar
        )

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        decoded = self.mlp(latent)
        return decoded[..., 0], decoded[..., 1]
