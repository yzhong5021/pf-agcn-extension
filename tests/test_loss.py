import torch

from modules.loss import BCEWithLogits


def test_matches_torch() -> None:
    torch.manual_seed(0)
    criterion = BCEWithLogits(pos_weight=None)
    logits = torch.tensor([[0.5, -1.0], [1.0, 0.0]], dtype=torch.float32)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    loss = criterion(logits, targets)

    manual = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="mean"
    )

    assert torch.isclose(loss, manual)


def test_pos_weight() -> None:
    logits = torch.zeros(1, 1)
    targets = torch.ones(1, 1)

    balanced = BCEWithLogits(pos_weight=None)(logits, targets)
    weighted = BCEWithLogits(pos_weight=torch.tensor([2.0]))(logits, targets)

    assert weighted > balanced
