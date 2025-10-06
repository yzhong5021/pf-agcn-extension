"""test_adaptive_protein.py

Minimal checks for AdaptiveProteinBlock.
"""

import torch

from modules.adaptive_protein import AdaptiveProteinBlock


def test_forward_with_prior():
    torch.manual_seed(0)
    block = AdaptiveProteinBlock(d_in=10, d_attn=10, steps=1, p=0.75)
    feats = torch.randn(5, 10)
    prior = torch.rand(5, 5)

    out = block(feats, prior)

    assert out.shape == feats.shape
    assert torch.isfinite(out).all()


def test_forward_without_prior():
    block = AdaptiveProteinBlock(d_in=6, d_attn=6, steps=2, p=0.8)
    feats = torch.randn(4, 6)

    out = block(feats)

    assert out.shape == feats.shape
    assert torch.isfinite(out).all()
