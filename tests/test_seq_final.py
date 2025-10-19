import torch

from src.modules.seq_final import SeqFinal


def test_seq_final_prot_proj_shape() -> None:
    torch.manual_seed(0)
    module = SeqFinal(in_dim=6, N_C=3, proj=4, out_ch=5)
    proteins = torch.randn(4, 6)

    out = module.prot_proj(proteins)

    assert out.shape == (4, 4, 5)
    assert torch.isfinite(out).all()


def test_seq_final_go_proj_shape() -> None:
    torch.manual_seed(1)
    module = SeqFinal(in_dim=5, N_C=2, proj=3, out_ch=4)
    proteins = torch.randn(3, 5)

    out = module.go_proj(proteins)

    assert out.shape == (2, 3, 4)
    assert torch.isfinite(out).all()
