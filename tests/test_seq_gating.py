"""test_seq_gating.py

Minimal checks for SeqGating fusion.
"""

import torch

from modules.seq_gating import SeqGating


def test_forward_shape():
    torch.manual_seed(0)
    gate = SeqGating(d_shared=32, d_esm=48, c_dcc=32, attn_hidden=16, dropout=0.0)
    esm = torch.randn(3, 20, 48)
    dcc = torch.randn(3, 20, 32)
    lengths = torch.tensor([20, 18, 15])

    fused = gate(esm, dcc, lengths)

    assert fused.shape == (3, 32)
    assert torch.isfinite(fused).all()
