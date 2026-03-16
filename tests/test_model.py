from __future__ import annotations

import torch

from battery_predict.models import LatentCapacityPredictor
from battery_predict.training.config import (
    DecoderConfig,
    EncoderConfig,
    PredictorConfig,
)
from battery_predict.training.losses import gaussian_nll, masked_mse


def test_model_forward_shapes() -> None:
    model = LatentCapacityPredictor(
        EncoderConfig(
            d_model=32,
            latent_dim=32,
            conv_channels=(16, 32),
            conv_kernels=(5, 3),
            conv_strides=(1, 2),
            transformer_layers=1,
            attention_heads=4,
            ff_dim=64,
            pooling_heads=2,
            max_signal_positions=64,
        ),
        PredictorConfig(
            d_model=32, layers=2, attention_heads=4, ff_dim=64, dropout=0.1
        ),
        DecoderConfig(hidden_dim=32),
    )
    signals = torch.randn(2, 4, 21, 2)
    signal_mask = torch.ones(2, 4, 21, dtype=torch.bool)
    signal_mask[:, :, -3:] = False
    sequence_mask = torch.tensor([[True, True, True, True], [True, True, True, False]])

    outputs = model(signals, signal_mask, sequence_mask)

    assert outputs["latents"].shape == (2, 4, 32)
    assert outputs["predicted_next_latent"].shape == (2, 3, 32)
    assert outputs["capacity_mean"].shape == (2, 3)
    assert outputs["capacity_logvar"].shape == (2, 3)


def test_masked_losses_ignore_invalid_positions() -> None:
    pred = torch.tensor([[[0.0, 1.0], [3.0, 4.0]]])
    target = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]])
    mask = torch.tensor([[True, False]])
    mse = masked_mse(pred, target, mask)
    assert torch.isclose(mse, torch.tensor(0.0))

    mean = torch.tensor([[0.0, 5.0]])
    logvar = torch.tensor([[0.0, 0.0]])
    capacity_target = torch.tensor([[0.0, 100.0]])
    nll, clamped = gaussian_nll(
        mean,
        logvar,
        capacity_target,
        mask,
        logvar_min=-10.0,
        logvar_max=3.0,
        eps=1e-6,
    )
    assert torch.isfinite(nll)
    assert torch.equal(clamped, logvar)
