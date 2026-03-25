"""Adaptive graph filtering layer"""

from typing import Optional
import torch.nn as nn, torch
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange


class AGFAttention(nn.Module):
    """
    Adaptive Graph Filtering Layer with OV Circuit and Multi-Head support.

    Flow:
    1. Estimator computes Adjacency A from Q, K.
    2. Input projected to Values V.
    3. Filter computes X' = Poly(A, V).
    4. Heads concatenated and projected via Output Linear.
    """

    _act_map = {
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
        "relu": F.relu,
        "gelu": F.gelu,
        "softmax": lambda x: F.softmax(x, dim=0),
        "identity": lambda x: x,
        "none": lambda x: x,
    }

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_head: int = 64,
        order: int = 3,
        top_k: Optional[int] = None,
        basis: str = "monomial",
        alphas_act: str = "sigmoid",
        max_relative_position: Optional[int] = None,
        normalization: str = "softmax",
    ):
        super().__init__()
        self.num_heads, self.dim_head, self.order, self.k = (
            num_heads,
            dim_head,
            order,
            top_k,
        )
        self.normalization = normalization
        self.order += 1  # Include the 0-th order term
        self.max_relative_position = max_relative_position
        self.alphas_act_name = alphas_act.lower()
        self.basis, self.scale = basis.lower(), dim_head**-0.5

        inner = num_heads * dim_head
        self.to_qkv = nn.Linear(dim, inner * 3, bias=True)
        self.to_out = nn.Linear(inner, dim, bias=True)

        if self.max_relative_position is not None and self.max_relative_position > 0:
            vocab_size = 2 * self.max_relative_position + 1
            self.relative_position_k = nn.Embedding(vocab_size, dim_head)
            self.relative_position_v = nn.Embedding(vocab_size, dim_head)

        # Coefficients
        self.alphas_raw = nn.Parameter(torch.empty(self.order, num_heads))
        self.act = self._act_map.get(alphas_act.lower(), lambda x: x)

        self.register_buffer("last_adj", None, persistent=False)
        self.register_buffer("last_x", None, persistent=False)
        self.reset_parameters()

    @property
    def alphas(self):
        return self.act(self.alphas_raw)

    def _build_adj(self, attn_scores):
        """Builds adjacency matrix from raw attention scores."""
        if self.normalization == "gelu":
            A = F.gelu(attn_scores)
            # Degree-normalise: D^{-1} A  (directed random walk)
            deg = A.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            return A / deg
        else:
            return attn_scores.softmax(dim=-1)

    @torch.no_grad()
    def reset_parameters(self):
        """
        Initializes alpha coefficients with a decay factor to prevent
        vanishing gradients and focus initially on low-order terms.
        """
        indices = torch.arange(self.order, device=self.alphas_raw.device).float()

        if self.alphas_act_name == "softmax":
            init_values = torch.exp(-indices)  # [1.0, 0.36, 0.13, ...]
        elif self.alphas_act_name in ["sigmoid", "tanh"]:
            if self.normalization == "softmax":
                init_values = torch.zeros(self.order)
            else:
                init_values = 0.5 * torch.exp(-indices)
        else:
            init_values = torch.ones(self.order) / self.order

        init_values = init_values.unsqueeze(-1).repeat(1, self.num_heads)
        self.alphas_raw.copy_(init_values)

        return self

    def forward(self, x, mask=None, is_causal: bool = False):
        b, n, _ = x.shape
        h = self.num_heads
        self.last_x = x
        # Unpack: (B, N, 3*H*D) -> 3 * (B, H, N, D)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv
        )

        attn_scores: Tensor = (q @ k.transpose(-1, -2)) * self.scale

        if self.max_relative_position:
            coords = torch.arange(n, device=x.device)
            distance = coords[:, None] - coords[None, :]  # Shape: (N, N)
            distance = torch.clamp(
                distance, -self.max_relative_position, self.max_relative_position
            )
            distance = distance + self.max_relative_position

            rel_k = self.relative_position_k(distance)  # (N, N, D)

            rel_k_scores = torch.einsum("bhid,ijd->bhij", q, rel_k)
            attn_scores = attn_scores + rel_k_scores

        if is_causal and mask is None:
            causal_mask = torch.ones((n, n), device=x.device, dtype=torch.bool).tril()
            attn_scores = attn_scores.masked_fill(~causal_mask, float("-inf"))
        elif mask is not None:
            mask_bc = mask.view(b, 1, 1, n)
            if mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(~mask_bc, float("-inf"))
            else:
                attn_scores = attn_scores.masked_fill(mask_bc == 0, float("-inf"))

        if self.k and self.k < n:
            top_val = attn_scores.topk(self.k, dim=-1)[0][..., -1:]
            attn_scores = attn_scores.masked_fill(attn_scores < top_val, float("-inf"))

        attn = self._build_adj(attn_scores)
        self.last_adj, self.v = attn.detach(), v.detach()

        with torch.autocast(device_type=x.device.type, enabled=False):
            attn, v = attn.float(), v.float()
            alphas = self.alphas.view(self.order, 1, self.num_heads, 1, 1).float()

            if self.max_relative_position:
                rel_v = self.relative_position_v(distance).float()

            res = alphas[0] * v
            v_prev, v_curr = None, v
            v_norm = v.norm(dim=-1, keepdim=True)

            for i in range(1, self.order):
                a_v = attn @ v_curr

                if i == 1 and self.max_relative_position:
                    a_v = a_v + torch.einsum("bhij,ijd->bhid", attn, rel_v)

                current_norm = a_v.norm(dim=-1, keepdim=True)
                a_v = a_v * (v_norm / (current_norm + 1e-6))

                if self.basis == "monomial":
                    v_curr = a_v
                else:
                    # Chebyshev Recurrence: T_k = 2(2A - I)T_{k-1} - T_{k-2}
                    l_v = 2 * a_v - v_curr
                    v_next = (2 * l_v - v_prev) if v_prev is not None else l_v
                    v_prev, v_curr = v_curr, v_next

                res = res + alphas[i] * v_curr

        out = res.to(x.dtype).transpose(1, 2).contiguous().view(b, n, -1)
        return self.to_out(out)

    def get_reg_loss(self):
        """Computes Spectral Smoothness Loss"""
        if self.last_adj is None:
            return torch.tensor(0.0, device=self.to_out.weight.device)

        # Dirichlet Energy on Normalized Features
        adj = self.last_adj.float()
        x_n = F.normalize(self.last_x.detach().float(), dim=-1)

        # "bnd,bhnm,bmd->" calculates Tr(X^T L X) efficiently
        smooth = torch.einsum("bnd,bhnm,bmd->", x_n, adj, x_n)

        bs, heads, n, _ = adj.shape
        return -smooth / (bs * heads * n)
