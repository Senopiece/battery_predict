from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def choose_group_count(channels: int, requested_groups: int) -> int:
    groups = min(channels, requested_groups)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


def downsample_mask(
    mask: torch.Tensor,
    *,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int = 1,
) -> torch.Tensor:
    if mask.dtype != torch.float32:
        mask_float = mask.to(dtype=torch.float32)
    else:
        mask_float = mask
    mask_float = mask_float.unsqueeze(1)
    kernel = torch.ones((1, 1, kernel_size), device=mask.device, dtype=mask_float.dtype)
    covered = F.conv1d(
        mask_float,
        kernel,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    return covered.squeeze(1) > 0


class SinusoidalPositionalEncoding(nn.Module):
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


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ff_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
