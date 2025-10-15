import pytest
import torch

from modules.dccn import DCCN_1D


def test_forward_shape() -> None:
    torch.manual_seed(0)
    block = DCCN_1D(embed_len=16, k_size=3, dilation=2, dropout=0.0)
    tokens = torch.randn(2, 12, 16)

    out = block(tokens)

    assert out.shape == tokens.shape
    assert torch.isfinite(out).all()


def test_backward_grad() -> None:
    block = DCCN_1D(embed_len=16)
    tokens = torch.randn(1, 8, 16, requires_grad=True)

    block(tokens).sum().backward()

    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()


def test_bad_rank() -> None:
    block = DCCN_1D(embed_len=8)
    bad_tokens = torch.randn(8, 8)

    with pytest.raises(IndexError):
        block(bad_tokens)
