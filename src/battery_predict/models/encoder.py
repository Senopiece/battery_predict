from __future__ import annotations

import torch
from torch import nn

from battery_predict.models.agf import AGFAttention
from battery_predict.models.embeddings import SinusoidalPositionalEncoding
from battery_predict.models.layers import (
    FeedForward,
    MaskedAttentionPooling,
    choose_group_count,
    downsample_mask,
)
from battery_predict.training.config import EncoderConfig


class ConvFeatureExtractor(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        if not (
            len(config.conv_channels)
            == len(config.conv_kernels)
            == len(config.conv_strides)
        ):
            raise ValueError(
                "Convolution channel, kernel, and stride tuples must align."
            )

        layers = []
        in_channels = 2  # Always 2 input channels: voltage, current
        self.spec: list[tuple[int, int, int, int]] = []
        for out_channels, kernel_size, stride in zip(
            config.conv_channels,
            config.conv_kernels,
            config.conv_strides,
            strict=True,
        ):
            padding = kernel_size // 2
            layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            layers.append(
                nn.GroupNorm(
                    choose_group_count(out_channels, config.conv_group_norm_groups),
                    out_channels,
                )
            )
            layers.append(nn.GELU())
            self.spec.append((kernel_size, stride, padding, 1))
            in_channels = out_channels
        self.layers = nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = x.transpose(1, 2)
        current_mask = mask
        spec_iter = iter(self.spec)

        for layer in self.layers:
            hidden = layer(hidden)
            if isinstance(layer, nn.Conv1d):
                kernel_size, stride, padding, dilation = next(spec_iter)
                current_mask = downsample_mask(
                    current_mask,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                )
                hidden = hidden * current_mask.unsqueeze(1).to(hidden.dtype)

        return hidden.transpose(1, 2), current_mask


class SignalTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        agf_order: int,
        agf_top_k: int | None,
        agf_alphas_act: str,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = AGFAttention(
            dim=d_model,
            num_heads=num_heads,
            dim_head=d_model // num_heads,
            order=agf_order,
            top_k=agf_top_k,
            basis="monomial",
            alphas_act=agf_alphas_act,
            max_relative_position=None,
            normalization="softmax",
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, ff_dim, dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attn_input = self.norm1(x)
        attn_out = self.attn(
            attn_input,
            mask=mask,
        )
        x = x + self.dropout1(attn_out)
        x = x * mask.unsqueeze(-1).to(x.dtype)
        x = x + self.dropout2(self.ff(self.norm2(x)))
        x = x * mask.unsqueeze(-1).to(x.dtype)
        return x


class CycleEncoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.conv = ConvFeatureExtractor(config)
        self.position = SinusoidalPositionalEncoding(
            config.d_model,
        )
        self.blocks = nn.ModuleList(
            [
                SignalTransformerBlock(
                    d_model=config.d_model,
                    num_heads=config.attention_heads,
                    ff_dim=config.ff_dim,
                    dropout=config.dropout,
                    agf_order=config.agf_order,
                    agf_top_k=config.agf_top_k,
                    agf_alphas_act=config.agf_alphas_act,
                )
                for _ in range(config.transformer_layers)
            ]
        )
        self.pool = MaskedAttentionPooling(config.d_model, config.pooling_heads)
        self.project = nn.Sequential(
            nn.Linear(config.d_model * config.pooling_heads, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
        )

    def forward(self, signal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden, hidden_mask = self.conv(signal, mask)
        hidden = self.position(hidden)
        hidden = hidden * hidden_mask.unsqueeze(-1).to(hidden.dtype)
        for block in self.blocks:
            hidden = block(hidden, hidden_mask)
        pooled = self.pool(hidden, hidden_mask)
        return self.project(pooled)
