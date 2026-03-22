"""Positional and rotary embeddings for sequence modeling."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalEmbedding(nn.Module):
    """Fixed sinusoidal embedding for arbitrary integer offsets.

    Maps (n,) long offsets → (n, dim) float embeddings on the fly.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.pow(
            10000.0,
            -torch.arange(half, dtype=torch.float32) / half,
        )
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, offsets: torch.Tensor) -> torch.Tensor:
        """offsets: (n,) long/int → (n, dim) float."""
        angles = offsets.float().unsqueeze(1) * self.freqs.unsqueeze(0)
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return emb[:, : self.dim]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1)
    return rotated.flatten(start_dim=-2)


class RotaryPositionalEncoding(nn.Module):
    """Rotary positional encoding (RoPE) applied additively to sequences."""

    def __init__(self, d_model: int, base: float = 10000.0, max_positions: int = 64):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("Rotary embedding requires an even dimension.")
        self.d_model = d_model
        inv_freq = 1.0 / (
            base ** (torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_positions)

    def _build_cache(self, max_positions: int) -> None:
        positions = torch.arange(
            max_positions,
            dtype=self.inv_freq.dtype,
            device=self.inv_freq.device,
        )
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len, d_model) → (B, seq_len, d_model) with rotary encoding."""
        cos, sin = self.get_cos_sin(x.size(1), device=x.device, dtype=x.dtype)
        return x * cos.unsqueeze(0) + rotate_half(x) * sin.unsqueeze(0)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding following the Transformer architecture.

    Builds the cache on the fly and extends it as needed.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term, persistent=False)
        self._build_cache(64)

    def _build_cache(self, length: int) -> None:
        position = torch.arange(
            length,
            dtype=self.div_term.dtype,
            device=self.div_term.device,
        ).unsqueeze(1)
        pe = torch.zeros(
            length,
            self.d_model,
            dtype=self.div_term.dtype,
            device=self.div_term.device,
        )
        pe[:, 0::2] = torch.sin(position * self.div_term)
        pe[:, 1::2] = torch.cos(position * self.div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def _ensure_cache(self, length: int) -> None:
        if length <= self.pe.size(1):
            return
        self._build_cache(length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_cache(x.size(1))
        return x + self.pe[:, : x.size(1)]
