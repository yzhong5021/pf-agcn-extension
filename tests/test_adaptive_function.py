import pytest
import torch

from modules.adaptive_function import AdaptiveFunctionBlock


def test_forward_prior() -> None:
    torch.manual_seed(0)
    block = AdaptiveFunctionBlock(d_in=8, d_attn=8, steps=1, p=0.8)
    feats = torch.randn(4, 8)
    prior = torch.rand(4, 4)

    out = block(feats, prior)

    assert out.shape == feats.shape
    assert torch.isfinite(out).all()


def test_forward_free() -> None:
    torch.manual_seed(7)
    block = AdaptiveFunctionBlock(d_in=6, d_attn=6, steps=2, p=0.9)
    feats = torch.randn(3, 6)

    out = block(feats)

    assert out.shape == feats.shape
    assert torch.isfinite(out).all()


def test_bad_prior() -> None:
    block = AdaptiveFunctionBlock(d_in=4, steps=1)
    feats = torch.randn(4, 4)
    bad_prior = torch.rand(4, 3)

    with pytest.raises(RuntimeError):
        block(feats, bad_prior)


def test_adj_rowsum() -> None:
    torch.manual_seed(11)
    block = AdaptiveFunctionBlock(d_in=5, d_attn=5, steps=1, p=0.7)
    feats = torch.randn(6, 5)

    adj = block._adj_from_feats(feats)

    row_sums = adj.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
