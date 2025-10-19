import torch

from src.modules.head import ClassificationHead


def test_classification_head_shapes() -> None:
    torch.manual_seed(0)
    head = ClassificationHead(N_C=5, d_in=6, dropout=0.0)
    functions = torch.randn(5, 6)
    proteins = torch.randn(7, 6)

    logits = head(functions, proteins)

    assert logits.shape == (7, 5)
    assert torch.isfinite(logits).all()
    assert head.bias.shape == (5,)
    assert head.log_tau.shape == torch.Size([])


def test_classification_head_scaling_changes_output() -> None:
    torch.manual_seed(1)
    head = ClassificationHead(N_C=3, d_in=4)
    functions = torch.randn(3, 4)
    proteins = torch.randn(2, 4)

    baseline = head(functions, proteins)
    head.log_tau.data.fill_(0.5)
    scaled = head(functions, proteins)

    assert not torch.allclose(baseline, scaled)
