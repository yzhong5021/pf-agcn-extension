"""test_seq_final.py

Minimal checks for SeqFinal projections.
"""

import torch

from modules.seq_final import SeqFinal


def test_go_proj_shape():
    torch.manual_seed(0)
    final = SeqFinal(in_dim=32, N_C=4, proj=8, out_ch=6)
    pooled = torch.randn(5, 32)

    go = final.go_proj(pooled)

    assert go.shape == (4, 5, 6)
    assert torch.isfinite(go).all()
