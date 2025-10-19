import torch
import pytest

from src.utils.prot_prior import (
    ESMProjector,
    generate_prior,
    prot_prior_blast,
    prot_prior_esm,
)


def test_prot_prior_esm_mean_reduce() -> None:
    torch.manual_seed(0)
    embeddings = torch.randn(3, 4, 8)
    projector = ESMProjector(in_dim=8, out_dim=6)

    prior = prot_prior_esm(embeddings, projector=projector, reduce="mean")

    assert prior.shape == (3, 3)
    assert torch.allclose(prior, prior.T, atol=1e-6)
    assert torch.allclose(prior.diag(), torch.zeros(3), atol=1e-6)


def test_prot_prior_esm_cls_reduce() -> None:
    torch.manual_seed(1)
    embeddings = torch.randn(2, 5, 4)

    prior = prot_prior_esm(embeddings, reduce="cls")

    assert prior.shape == (2, 2)
    assert torch.allclose(prior, prior.T, atol=1e-6)


def test_prot_prior_esm_invalid_reduce() -> None:
    embeddings = torch.randn(1, 2, 3)
    with pytest.raises(ValueError):
        prot_prior_esm(embeddings, reduce="median")


def test_generate_prior_invalid_method() -> None:
    with pytest.raises(ValueError):
        generate_prior(torch.randn(2, 2, 2), method="unknown")


def test_generate_prior_blast_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("Bio")

    class DummyBlast:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self):
            return "1e-06\n", ""

    monkeypatch.setattr("src.utils.prot_prior.NcbiblastpCommandline", DummyBlast)

    seqs = ["ACDE", "ACDF"]
    prior = prot_prior_blast(seqs, evalue_threshold=1e-3)

    assert prior.shape == (2, 2)
    assert prior[0, 1] == prior[1, 0] == 1.0
