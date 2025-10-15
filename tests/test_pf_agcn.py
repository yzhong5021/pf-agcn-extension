import pytest
import torch

from model.config import (
    AdaptiveFunctionConfig,
    AdaptiveProteinConfig,
    DCCNConfig,
    HeadConfig,
    LossConfig,
    PFAGCNConfig,
    PFAGCNModelConfig,
    PriorConfig,
    SeqFinalConfig,
    SeqGatingConfig,
    SequenceEmbeddingsConfig,
    TaskConfig,
)
from model.model import PFAGCN


def _build_config() -> PFAGCNConfig:
    model_cfg = PFAGCNModelConfig(
        seq_embeddings=SequenceEmbeddingsConfig(feature_dim=24, seq_len=32),
        prot_prior=PriorConfig(top_p_mass=0.8, temperature=0.9),
        go_prior=PriorConfig(top_p_mass=0.8, temperature=0.9),
        dccn=DCCNConfig(channels=24, kernel_size=3, dilation=1, dropout=0.0),
        seq_gating=SeqGatingConfig(shared_dim=16, esm_dim=24, dccn_channels=24, attn_hidden=8, dropout=0.0),
        seq_final=SeqFinalConfig(metric_dim=12, graph_dim=16),
        adaptive_protein=AdaptiveProteinConfig(feature_dim=16, attention_dim=16, steps=1, top_p_mass=0.7, temperature=1.1, dropout=0.0),
        adaptive_function=AdaptiveFunctionConfig(feature_dim=16, attention_dim=16, steps=1, top_p_mass=0.7, temperature=1.1, dropout=0.0),
        head=HeadConfig(feature_dim=16, dropout=0.0),
        loss=LossConfig(name="bce_with_logits"),
    )
    return PFAGCNConfig(
        task=TaskConfig(num_functions=5, vocab_size=21, pad_index=0),
        model=model_cfg,
    )


def _build_model() -> PFAGCN:
    return PFAGCN(_build_config())


def _sample_embeddings(batch: int = 3, length: int = 32, dim: int = 24) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(batch, length, dim)


def test_forward_shape() -> None:
    model = _build_model()
    embeddings = _sample_embeddings()
    lengths = torch.full((embeddings.size(0),), embeddings.size(1), dtype=torch.long)

    output = model(embeddings, lengths=lengths)

    assert output.logits.shape == (embeddings.size(0), 5)
    assert output.protein_embeddings.shape == (embeddings.size(0), 16)
    assert output.function_embeddings.shape == (5, 16)
    assert torch.isfinite(output.logits).all()


def test_bad_embedding_rank() -> None:
    model = _build_model()
    embeddings = torch.randn(2, 3, 4, 5)
    lengths = torch.full((2,), 3, dtype=torch.long)

    with pytest.raises(ValueError):
        model(embeddings, lengths=lengths)


def test_bad_length_rank() -> None:
    model = _build_model()
    embeddings = _sample_embeddings(batch=2)
    lengths = torch.full((2, 1), embeddings.size(1), dtype=torch.long)

    with pytest.raises(ValueError):
        model(embeddings, lengths=lengths)


def test_bad_protein_prior_shape() -> None:
    model = _build_model()
    embeddings = _sample_embeddings(batch=2)
    lengths = torch.full((2,), embeddings.size(1), dtype=torch.long)
    bad_prior = torch.rand(3, 3)

    with pytest.raises(ValueError):
        model(embeddings, lengths=lengths, protein_prior=bad_prior)
