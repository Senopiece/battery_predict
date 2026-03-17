from __future__ import annotations

import math

import torch


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(values.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return (values * mask_f).sum() / denom


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    error = (prediction - target).pow(2).mean(dim=-1)
    return masked_mean(error, mask)


def masked_mse_scalar(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    error = (prediction - target).pow(2)
    return masked_mean(error, mask)


def gaussian_nll(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    logvar_min: float,
    logvar_max: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    clamped_logvar = logvar.clamp(min=logvar_min, max=logvar_max)
    variance = torch.exp(clamped_logvar).clamp_min(eps)
    nll = 0.5 * (
        ((target - mean).pow(2) / variance) + clamped_logvar + math.log(2.0 * math.pi)
    )
    return masked_mean(nll, mask), clamped_logvar
