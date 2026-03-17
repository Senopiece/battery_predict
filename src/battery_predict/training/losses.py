from __future__ import annotations

import torch

_LOG_2PI = 0.9189385332046727  # log(2*pi)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(values.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return (values * mask_f).sum() / denom


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    error = (prediction - target).square().mean(dim=-1)
    return masked_mean(error, mask)


def masked_mse_scalar(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    error = (prediction - target).square()
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
    nll = 0.5 * (((target - mean).square() / variance) + clamped_logvar + _LOG_2PI)
    return masked_mean(nll, mask), clamped_logvar
