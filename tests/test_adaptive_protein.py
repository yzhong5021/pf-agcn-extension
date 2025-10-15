import pytest
import torch

from modules.adaptive_protein import AdaptiveProteinBlock


def test_forward_prior() -> None:
    torch.manual_seed(0)
    block = AdaptiveProteinBlock(d_in=10, d_attn=10, steps=1, p=0.75)
    feats = torch.randn(5, 10)
    prior = torch.rand(5, 5)

    out = block(feats, prior)

    assert out.shape == feats.shape
    assert torch.isfinite(out).all()


def test_forward_free() -> None:
    torch.manual_seed(13)
    block = AdaptiveProteinBlock(d_in=6, d_attn=6, steps=2, p=0.85)
    feats = torch.randn(4, 6)

    out = block(feats)

    assert out.shape == feats.shape
    assert torch.isfinite(out).all()


def test_bad_prior() -> None:
    block = AdaptiveProteinBlock(d_in=4, steps=1)
    feats = torch.randn(4, 4)
    bad_prior = torch.rand(4, 3)

    with pytest.raises(RuntimeError):
        block(feats, bad_prior)


def test_adj_rowsum() -> None:
    torch.manual_seed(2)
    block = AdaptiveProteinBlock(d_in=5, d_attn=5, steps=1, p=0.8)
    feats = torch.randn(6, 5)

    adj = block._adj_from_feats(feats)

    row_sums = adj.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
