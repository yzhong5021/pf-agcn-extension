"""test_pf_agcn.py

Smoke test for PFAGCN forward pass.
"""

import torch

from modules.mock_esm import MockESM
from model.pf_agcn import PFAGCN


def test_forward_shapes():
    torch.manual_seed(0)
    encoder = MockESM(seq_len=32, hidden_len=48, embed_len=24, proj_len=24)
    model = PFAGCN(
        num_functions=5,
        seq_encoder=encoder,
        seq_encoder_dim=24,
        shared_dim=24,
        graph_dim=16,
        metric_dim=12,
        dccn_channels=24,
        protein_steps=1,
        function_steps=1,
    )
    tokens = torch.randint(0, 21, (3, 32))
    lengths = torch.full((3,), 32, dtype=torch.long)

    output = model(tokens, lengths)

    assert output.logits.shape == (3, 5)
    assert output.protein_embeddings.shape == (3, 16)
    assert output.function_embeddings.shape == (5, 16)
    assert torch.isfinite(output.logits).all()
