import pytest
import torch

from src.modules.adaptive_protein import AdaptiveProteinBlock


def test_adaptive_protein_forward_with_prior() -> None:
    torch.manual_seed(0)
    block = AdaptiveProteinBlock(d_in=10, d_attn=6, steps=2, p=0.8)
    x = torch.randn(4, 10)
    prior = torch.rand(4, 4)

    out = block(x, prior)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_adaptive_protein_forward_identity_prior() -> None:
    torch.manual_seed(1)
    block = AdaptiveProteinBlock(d_in=5, d_attn=4, steps=1, p=0.7)
    x = torch.randn(3, 5)

    out = block(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_adaptive_protein_bad_prior_shape() -> None:
    block = AdaptiveProteinBlock(d_in=3, steps=1)
    x = torch.randn(2, 3)
    bad_prior = torch.rand(3, 3)

    with pytest.raises(RuntimeError):
        block(x, bad_prior)
