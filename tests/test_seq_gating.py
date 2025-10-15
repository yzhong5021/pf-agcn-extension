import pytest
import torch

from modules.seq_gating import SeqGating


def test_forward_shape() -> None:
    torch.manual_seed(0)
    gate = SeqGating(
        d_shared=32,
        d_esm=48,
        c_dcc=32,
        attn_hidden=16,
        dropout=0.0,
    )
    esm = torch.randn(3, 20, 48)
    dcc = torch.randn(3, 20, 32)
    lengths = torch.tensor([20, 18, 15])

    fused = gate(esm, dcc, lengths)

    assert fused.shape == (3, 32)
    assert torch.isfinite(fused).all()


def test_padding_mask() -> None:
    torch.manual_seed(2)
    gate = SeqGating(
        d_shared=16,
        d_esm=16,
        c_dcc=16,
        attn_hidden=8,
        dropout=0.0,
    )
    esm = torch.randn(1, 5, 16)
    dcc = torch.randn(1, 5, 16)
    lengths = torch.tensor([3])

    fused = gate(esm, dcc, lengths)

    esm_perturbed = esm.clone()
    dcc_perturbed = dcc.clone()
    esm_perturbed[:, lengths[0] :] += torch.randn(1, 2, 16)
    dcc_perturbed[:, lengths[0] :] += torch.randn(1, 2, 16)

    fused_perturbed = gate(esm_perturbed, dcc_perturbed, lengths)

    assert torch.allclose(fused, fused_perturbed, atol=1e-5)


def test_dim_mismatch() -> None:
    gate = SeqGating(
        d_shared=8,
        d_esm=10,
        c_dcc=6,
        attn_hidden=4,
        dropout=0.0,
    )
    esm = torch.randn(2, 4, 10)
    dcc = torch.randn(2, 4, 5)
    lengths = torch.tensor([4, 4])

    with pytest.raises(RuntimeError):
        gate(esm, dcc, lengths)
