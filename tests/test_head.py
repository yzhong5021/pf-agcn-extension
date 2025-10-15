import pytest
import torch

from modules.head import ClassificationHead


def test_logits_shape() -> None:
    torch.manual_seed(0)
    head = ClassificationHead(N_C=7, d_in=12, dropout=0.0)
    function_feats = torch.randn(7, 12)
    protein_feats = torch.randn(4, 12)

    logits = head(function_feats, protein_feats)

    assert logits.shape == (4, 7)
    assert torch.isfinite(logits).all()


def test_tau_grad() -> None:
    head = ClassificationHead(N_C=3, d_in=6, dropout=0.0)
    function_feats = torch.randn(3, 6, requires_grad=True)
    protein_feats = torch.randn(2, 6, requires_grad=True)

    logits = head(function_feats, protein_feats)
    logits.sum().backward()

    assert head.log_tau.grad is not None
    assert torch.isfinite(head.log_tau.grad)


def test_dim_mismatch() -> None:
    head = ClassificationHead(N_C=3, d_in=4, dropout=0.0)
    function_feats = torch.randn(3, 5)
    protein_feats = torch.randn(2, 4)

    with pytest.raises(RuntimeError):
        head(function_feats, protein_feats)
