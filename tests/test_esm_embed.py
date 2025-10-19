﻿from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.utils import esm_embed


class DummyTokenizer:
    def __init__(self, store):
        self.cls_token_id = 0
        self.pad_token_id = 1
        self.eos_token_id = 2
        self.store = store

    def __call__(self, sequences, **kwargs):
        self.store["called_with"] = list(sequences)
        batch = torch.tensor([[0, 3, 2, 1], [0, 4, 2, 1]])
        return {"input_ids": batch, "attention_mask": torch.ones_like(batch)}


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.ones(1))

    def forward(self, **batch):
        input_ids = batch["input_ids"]
        bsz, seqlen = input_ids.shape
        hidden = torch.arange(seqlen, dtype=torch.float32).repeat(bsz, 1).view(bsz, seqlen, 1)
        hidden = hidden.expand(bsz, seqlen, 4)
        return SimpleNamespace(last_hidden_state=hidden)

    def eval(self):
        return self

    def to(self, device):
        return self


def test_esm_embed_get_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    store = {}

    monkeypatch.setattr(esm_embed.EsmTokenizer, "from_pretrained", lambda *args, **kwargs: DummyTokenizer(store))
    monkeypatch.setattr(esm_embed.EsmModel, "from_pretrained", lambda *args, **kwargs: DummyModel())

    embedder = esm_embed.ESM_Embed(model_name="dummy", max_len=5, truncate_len=3)

    hidden, mask = embedder.get_esm_embed(["ABCDEFG", "HI"])

    assert store["called_with"][0] == "ABC"
    assert hidden.shape == mask.shape + (4,)
    assert mask.dtype == torch.bool
    assert torch.equal(mask[0], torch.tensor([False, True, False, False]))
    assert all(not param.requires_grad for param in embedder.model.parameters())
