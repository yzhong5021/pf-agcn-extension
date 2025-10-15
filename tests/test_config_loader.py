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


def test_config_defaults() -> None:
    cfg = PFAGCNConfig()

    assert isinstance(cfg.task, TaskConfig)
    assert isinstance(cfg.model, PFAGCNModelConfig)

    model_cfg = cfg.model
    assert isinstance(model_cfg.seq_embeddings, SequenceEmbeddingsConfig)
    assert isinstance(model_cfg.prot_prior, PriorConfig)
    assert isinstance(model_cfg.go_prior, PriorConfig)
    assert isinstance(model_cfg.dccn, DCCNConfig)
    assert isinstance(model_cfg.seq_gating, SeqGatingConfig)
    assert isinstance(model_cfg.seq_final, SeqFinalConfig)
    assert isinstance(model_cfg.adaptive_protein, AdaptiveProteinConfig)
    assert isinstance(model_cfg.adaptive_function, AdaptiveFunctionConfig)
    assert isinstance(model_cfg.head, HeadConfig)
    assert isinstance(model_cfg.loss, LossConfig)


def test_custom_overrides() -> None:
    model_cfg = PFAGCNModelConfig(
        seq_embeddings=SequenceEmbeddingsConfig(feature_dim=512),
        prot_prior=PriorConfig(top_p_mass=0.8),
        adaptive_protein=AdaptiveProteinConfig(steps=4),
    )
    cfg = PFAGCNConfig(
        task=TaskConfig(num_functions=1024, vocab_size=25, pad_index=1),
        model=model_cfg,
    )

    assert cfg.task.num_functions == 1024
    assert cfg.model.seq_embeddings.feature_dim == 512
    assert cfg.model.prot_prior.top_p_mass == 0.8
    assert cfg.model.adaptive_protein.steps == 4
