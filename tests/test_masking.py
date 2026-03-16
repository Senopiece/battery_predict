from __future__ import annotations

import torch

from battery_predict.models.layers import downsample_mask


def test_downsample_mask_tracks_any_valid_value_in_receptive_field() -> None:
    mask = torch.tensor([[True, True, False, False, True, False, False]])
    downsampled = downsample_mask(mask, kernel_size=3, stride=2, padding=1)
    expected = torch.tensor([[True, True, True, False]])
    assert torch.equal(downsampled, expected)
