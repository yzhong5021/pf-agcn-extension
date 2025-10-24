"""Utilities for loading pretrained ESM embeddings for PF-AGCN."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import EsmModel, EsmTokenizer


class ESM_Embed(nn.Module):
    """Thin wrapper around HuggingFace ESM models with frozen weights."""

    def __init__(
        self,
        model_name: str = "facebook/esm1b_t33_650M_UR50S",
        max_len: int = 1022,
        truncate_len: int = 1000,
        device: str | torch.device = "cpu",
        cache_dir: Optional[str | Path] = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        pretrained_kwargs = {}
        if self.cache_dir is not None:
            pretrained_kwargs["cache_dir"] = str(self.cache_dir)
        self.tokenizer = EsmTokenizer.from_pretrained(model_name, **pretrained_kwargs)
        self.model = EsmModel.from_pretrained(model_name, **pretrained_kwargs)
        self.model.eval()
        self.model.to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.max_len = int(max_len)
        self.truncate_len = int(truncate_len)
        self.cls_id = self.tokenizer.cls_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id

    @torch.inference_mode()
    def get_esm_embed(self, seqs: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return residue-level embeddings and masks for provided sequences."""
        sequences = [seq if len(seq) <= self.max_len else seq[:self.truncate_len] for seq in seqs]

        batch = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=False, # truncation is done prior
            add_special_tokens=True,
            return_attention_mask=True,
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}

        outputs = self.model(**batch)
        input_ids = batch["input_ids"]
        residue_mask = (input_ids != self.pad_id) & (input_ids != self.cls_id) & (input_ids != self.eos_id)
        return outputs.last_hidden_state, residue_mask

    @torch.inference_mode()
    def forward(self, seqs: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """nn.Module forward passthrough for unified embedder interface."""
        return self.get_esm_embed(seqs)

    @torch.inference_mode()
    def get_embeddings(self, seqs: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias mirroring Prost wrapper interface."""
        return self.get_esm_embed(seqs)
