from __future__ import annotations

import torch
from torch import nn


class MaskedAttentionPooling(nn.Module):
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
