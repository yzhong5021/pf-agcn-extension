import pytest
import torch

from src.modules.adaptive_function import AdaptiveFunctionBlock


def test_adaptive_function_forward_shapes() -> None:
    torch.manual_seed(0)
    block = AdaptiveFunctionBlock(d_in=8, d_attn=4, steps=2, p=0.7)
    x = torch.randn(5, 8)

    out = block(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert not torch.allclose(out, x)


def test_adaptive_function_prior_and_attention_row_sum() -> None:
    torch.manual_seed(1)
    block = AdaptiveFunctionBlock(d_in=6, d_attn=4, steps=1, p=0.6)
    x = torch.randn(4, 6)
    prior = torch.rand(4, 4)

    out = block(x, prior)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    adj = block._adj_from_feats(x)
    row_sums = adj.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_adaptive_function_invalid_prior() -> None:
    block = AdaptiveFunctionBlock(d_in=4, steps=1)
    x = torch.randn(3, 4)
    bad_prior = torch.rand(3, 2)

    with pytest.raises(RuntimeError):
        block(x, bad_prior)
