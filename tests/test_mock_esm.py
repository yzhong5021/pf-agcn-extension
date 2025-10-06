"""test_mock_esm.py

Minimal checks for MockESM encoder.
"""

import torch

from modules.mock_esm import MockESM


def test_forward_shape():
    torch.manual_seed(0)
    encoder = MockESM(seq_len=32, hidden_len=48, embed_len=24, proj_len=16)
    tokens = torch.randint(0, 21, (3, 32))

    output = encoder(tokens)

    assert output.shape == (3, 32, 16)
    assert torch.isfinite(output).all()
