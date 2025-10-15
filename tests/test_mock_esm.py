import pytest
import torch

from modules.mock_esm import MockESM


def test_forward_shape() -> None:
    torch.manual_seed(0)
    encoder = MockESM(seq_len=32, hidden_len=48, embed_len=24, proj_len=16)
    tokens = torch.randint(0, 21, (3, 32))

    output = encoder(tokens)

    assert output.shape == (3, 32, 16)
    assert torch.isfinite(output).all()


def test_backward_grad() -> None:
    torch.manual_seed(4)
    encoder = MockESM(seq_len=16, hidden_len=20, embed_len=12, proj_len=12)
    tokens = torch.randint(0, 21, (2, 16))

    out = encoder(tokens).sum()
    out.backward()

    assert encoder.embed.weight.grad is not None
    assert torch.isfinite(encoder.embed.weight.grad).all()


def test_bad_token() -> None:
    encoder = MockESM(seq_len=8, hidden_len=12, embed_len=6, proj_len=6)
    tokens = torch.full((2, 8), 99, dtype=torch.long)

    with pytest.raises(IndexError):
        encoder(tokens)
