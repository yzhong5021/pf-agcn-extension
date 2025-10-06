"""test_head.py

Quick checks for ClassificationHead.
"""

import torch

from modules.head import ClassificationHead


def test_logits_shape():
    torch.manual_seed(0)
    head = ClassificationHead(N_C=7, d_in=12, dropout=0.0)
    function_feats = torch.randn(7, 12)
    protein_feats = torch.randn(4, 12)

    logits = head(function_feats, protein_feats)

    assert logits.shape == (4, 7)
    assert torch.isfinite(logits).all()
