"""Reusable neural network layers and utilities."""

from __future__ import annotations
import math

import torch
from torch import nn
from torch.nn import functional as F


def choose_group_count(channels: int, requested_groups: int) -> int:
    """Choose the largest divisor of channels that doesn't exceed requested_groups."""
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
    """Downsample a binary mask through a Conv1d operation."""
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


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

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


class MaskedAttentionPooling(nn.Module):
    """Multi-head attention-based pooling for sequences."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.score_proj = nn.Linear(d_model, d_model)
        self.score_out = nn.Linear(d_model, num_heads)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score_out(torch.tanh(self.score_proj(hidden)))
        scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        weights = torch.softmax(scores, dim=1)
        pooled = torch.einsum("bth,btd->bhd", weights, hidden)
        return pooled.flatten(start_dim=1)


class PositiveLinear(nn.Module):
    """Linear layer with positive weights and bias via softplus."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight_unconstrained = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias_unconstrained = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias_unconstrained", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_unconstrained, a=math.sqrt(5))
        if self.bias_unconstrained is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_unconstrained)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_unconstrained, -bound, bound)

    def forward(self, input):
        weight = F.softplus(self.weight_unconstrained)
        bias = (
            F.softplus(self.bias_unconstrained)
            if self.bias_unconstrained is not None
            else None
        )
        return F.linear(input, weight, bias)
