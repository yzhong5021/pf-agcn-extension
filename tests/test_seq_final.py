import torch

from modules.seq_final import SeqFinal


def test_go_shape() -> None:
    torch.manual_seed(0)
    final = SeqFinal(in_dim=32, N_C=4, proj=8, out_ch=6)
    pooled = torch.randn(5, 32)

    go = final.go_proj(pooled)

    assert go.shape == (4, 5, 6)
    assert torch.isfinite(go).all()


def test_prot_shape(capfd) -> None:
    torch.manual_seed(4)
    final = SeqFinal(in_dim=16, N_C=3, proj=4, out_ch=5)
    pooled = torch.randn(6, 16)

    prot = final.prot_proj(pooled)

    capfd.readouterr()

    assert prot.shape == (6, 6, 5)
    assert torch.isfinite(prot).all()
