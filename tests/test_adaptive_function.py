"""test_adaptive_function.py

Minimal checks for AdaptiveFunctionBlock.
"""

import torch

from modules.adaptive_function import AdaptiveFunctionBlock


def test_forward_with_prior():
    torch.manual_seed(0)
    block = AdaptiveFunctionBlock(d_in=8, d_attn=8, steps=1, p=0.8)
    feats = torch.randn(4, 8)
    prior = torch.rand(4, 4)

    out = block(feats, prior)

    assert out.shape == feats.shape
    assert torch.isfinite(out).all()


def test_forward_without_prior():
    block = AdaptiveFunctionBlock(d_in=6, d_attn=6, steps=1, p=0.9)
    feats = torch.zeros(3, 6)

    out = block(feats)

    assert out.shape == feats.shape
    assert torch.isfinite(out).all()
