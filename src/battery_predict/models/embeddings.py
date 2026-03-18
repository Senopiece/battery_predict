"""Positional and rotary embeddings for sequence modeling."""

from __future__ import annotations

import math

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1)
    return rotated.flatten(start_dim=-2)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding following the Transformer architecture."""

    def __init__(self, d_model: int, max_positions: int):
        super().__init__()
        position = torch.arange(max_positions, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_positions, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE) for efficient attention."""

    def __init__(self, head_dim: int, base: float = 10000.0, max_positions: int = 1024):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(
                "Rotary embedding requires an even attention head dimension."
            )
        self.head_dim = head_dim
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_positions)

    def _build_cache(self, max_positions: int) -> None:
        positions = torch.arange(max_positions, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _ensure_cache(self, steps: int) -> None:
        if steps <= self.cos_cached.size(0):
            return
        self._build_cache(steps)

    def get_cos_sin(
        self, steps: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_cache(steps)
        cos = self.cos_cached[:steps].to(device=device, dtype=dtype)
        sin = self.sin_cached[:steps].to(device=device, dtype=dtype)
        return cos, sin

    def apply(
        self, tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        return (tensor * cos) + (rotate_half(tensor) * sin)
