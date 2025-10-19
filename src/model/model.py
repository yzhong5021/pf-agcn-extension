"""PF-AGCN architecture definition.

Implements the dilated-conv → sequence projection → adaptive graph workflow
using cached sequence embeddings and priors supplied at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from model.config import PFAGCNConfig  # type: ignore
except ImportError:  # pragma: no cover
    from typing import Any as PFAGCNConfig  # type: ignore
from src.modules.adaptive_function import AdaptiveFunctionBlock
from src.modules.adaptive_protein import AdaptiveProteinBlock
from src.modules.dccn import DCCN_1D
from src.modules.head import ClassificationHead
from src.modules.seq_final import SeqFinal
from src.modules.seq_gating import SeqGating


@dataclass
class Output:
    """Container returned by the model forward pass."""

    logits: torch.Tensor
    protein_embeddings: torch.Tensor
    function_embeddings: torch.Tensor
    protein_adjacency: torch.Tensor
    function_adjacency: torch.Tensor


class PFAGCN(nn.Module):
    """Protein Function Adaptive Graph Convolutional Network.

    Complete architecture:
    Sequence embeddings -> DCCN -> ESM + DCCN representational gating -> adaptive protein/function blocks -> classification head
    """

    def __init__(self, config: PFAGCNConfig) -> None:
        super().__init__()
        self.config = config
        model_cfg = config.model
        self.num_functions = config.task.num_functions

        seq_dim = model_cfg.seq_embeddings.feature_dim
        dccn_channels = model_cfg.dccn.channels
        graph_dim = model_cfg.seq_final.graph_dim
        shared_dim = model_cfg.seq_gating.shared_dim
        self.use_protein_prior = bool(getattr(model_cfg.prot_prior, "enabled", False))
        self.protein_prior_method = getattr(model_cfg.prot_prior, "method", "cosine")
        self.use_go_prior = bool(getattr(model_cfg.go_prior, "enabled", True))

        if shared_dim != graph_dim:
            raise ValueError("seq_gating.shared_dim must equal seq_final.graph_dim")

        if seq_dim != dccn_channels:
            self.dccn_input = nn.Linear(seq_dim, dccn_channels, bias=False)
            nn.init.xavier_uniform_(self.dccn_input.weight)
        else:
            self.dccn_input = nn.Identity()

        self.dccn = DCCN_1D(
            embed_len=dccn_channels,
            k_size=model_cfg.dccn.kernel_size,
            dilation=model_cfg.dccn.dilation,
            dropout=model_cfg.dccn.dropout,
        )

        self.seq_gating = SeqGating(
            d_shared=shared_dim,
            d_esm=seq_dim,
            c_dcc=dccn_channels,
            attn_hidden=model_cfg.seq_gating.attn_hidden,
            dropout=model_cfg.seq_gating.dropout,
        )

        self.seq_final = SeqFinal(
            in_dim= shared_dim,
            N_C=self.num_functions,
            proj=model_cfg.seq_final.metric_dim,
            out_ch=graph_dim,
        )

        self.protein_block = AdaptiveProteinBlock(
            d_in=graph_dim,
            d_attn=model_cfg.adaptive_protein.attention_dim,
            steps=model_cfg.adaptive_protein.steps,
            p=model_cfg.adaptive_protein.top_p_mass,
            tau=model_cfg.adaptive_protein.temperature,
            dropout=model_cfg.adaptive_protein.dropout,
        )
        self.function_block = AdaptiveFunctionBlock(
            d_in=graph_dim,
            d_attn=model_cfg.adaptive_function.attention_dim,
            steps=model_cfg.adaptive_function.steps,
            p=model_cfg.adaptive_function.top_p_mass,
            tau=model_cfg.adaptive_function.temperature,
            dropout=model_cfg.adaptive_function.dropout,
        )
        self.head = ClassificationHead(
            N_C=self.num_functions,
            d_in=graph_dim,
            dropout=model_cfg.head.dropout,
        )

    def forward(
        self,
        seq_embeddings: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        protein_prior: Optional[torch.Tensor] = None,
        go_prior: Optional[torch.Tensor] = None,
    ) -> Output:
        """Run end-to-end inference with cached sequence embeddings."""

        if seq_embeddings.ndim != 3:
            raise ValueError("seq_embeddings must be a 3D tensor (batch, length, dim).")

        batch, seqlen, feat_dim = seq_embeddings.shape
        print("batch, seqlen, features:", [batch, seqlen, feat_dim])
        expected_dim = self.config.model.seq_embeddings.feature_dim
        if feat_dim != expected_dim:
            raise ValueError(
                f"Expected seq_embeddings feature dim {expected_dim}, got {feat_dim}."
            )

        if mask is None and lengths is None:
            raise ValueError("Either lengths or mask must be provided.")

        device = seq_embeddings.device
        mask_bool, lengths_tensor = self._normalise_mask(lengths, mask, batch, seqlen, device)

        if not self.use_go_prior:
            go_prior = None
        elif go_prior is None:
            log.debug("GO prior enabled but missing; proceeding without it for this batch.")

        embeddings_projected = self.dccn_input(seq_embeddings)
        print("\n\n\nEmbeddings_projected", embeddings_projected.shape)
        conv_features = self.dccn(embeddings_projected, mask=mask_bool)
        print("\n\n\nConv_features", conv_features.shape)

        
        gating_repr = self.seq_gating(
            H_esm=seq_embeddings,
            H_dcc=conv_features,
            lengths=lengths_tensor,
            mask=mask_bool,
        )

        print("\n\n\nGating_repr", gating_repr.shape)
        if self.use_protein_prior and protein_prior is None:
            protein_prior = self._build_protein_prior(gating_repr)
        elif not self.use_protein_prior:
            protein_prior = None


        protein_init, function_init = self._initial_graph_features(gating_repr)
        print("\n\n\nProtein_init (initial)", protein_init.shape)
        print("\n\n\nFunction_init (initial)", function_init.shape)

        # print("\n\n\nProtein prior", protein_prior.shape)
        protein_embeddings = self._run_protein_block(protein_init, protein_prior)
        print("\n\n\nProtein_embeddings", protein_embeddings.shape) 

        # print("\n\n\nGO prior", go_prior.shape)
        function_embeddings = self._run_function_block(function_init, go_prior)
        print("\n\n\nFunction_embeddings", function_embeddings.shape) 

        logits = self.head(function_embeddings, protein_embeddings)
        print("\n\n\nLogits", logits.shape) 

        protein_adj, function_adj = self._build_adjacencies(
            protein_embeddings, function_embeddings
        )
        print("\n\n\nProtein_adj", protein_adj.shape) 
        print("\n\n\nFunction_adj", function_adj.shape) 

        return Output(
            logits=logits,
            protein_embeddings=protein_embeddings,
            function_embeddings=function_embeddings,
            protein_adjacency=protein_adj,
            function_adjacency=function_adj,
        )

    def _normalise_mask(
        self,
        lengths: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        batch: int,
        seqlen: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask is not None:
            if mask.shape != (batch, seqlen):
                raise ValueError("mask must match the batch and sequence dimensions.")
            mask_bool = mask.to(device=device, dtype=torch.bool)
            lengths_tensor = mask_bool.sum(dim=1, dtype=torch.long)
            if lengths is not None:
                lengths = lengths.to(device=device, dtype=torch.long)
                if not torch.equal(lengths_tensor, lengths):
                    raise ValueError("mask and lengths encode conflicting values.")
        else:
            if lengths is None or lengths.ndim != 1:
                raise ValueError("lengths must be a 1D tensor when mask is absent.")
            lengths_tensor = lengths.to(device=device, dtype=torch.long)
            mask_bool = (
                torch.arange(seqlen, device=device)[None, :]
                < lengths_tensor[:, None]
            )
        return mask_bool, lengths_tensor

    def _masked_mean(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).to(dtype=tensor.dtype)
        total = (tensor * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return total / denom

    def _initial_graph_features(
        self, protein_repr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        protein_features = self.seq_final.prot_proj(protein_repr)
        function_features = self.seq_final.go_proj(protein_repr)
        return protein_features, function_features

    def _run_protein_block(
        self, protein_init: torch.Tensor, protein_prior: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if protein_prior is not None:
            if protein_prior.ndim != 2:
                raise ValueError("protein_prior must be 2D (N, N).")
            if protein_prior.shape[0] != protein_prior.shape[1]:
                raise ValueError("protein_prior must be square.")
            if protein_prior.shape[0] != protein_init.shape[0]:
                raise ValueError("protein_prior size must match batch size.")
            protein_prior = protein_prior.to(protein_init.device, dtype=protein_init.dtype)
        return self.protein_block(protein_init, protein_prior)

    def _run_function_block(
        self, function_init: torch.Tensor, go_prior: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if go_prior is not None:
            if go_prior.ndim != 2:
                raise ValueError("go_prior must be 2D (C, C).")
            if go_prior.shape[0] != go_prior.shape[1]:
                raise ValueError("go_prior must be square.")
            if go_prior.shape[0] != self.num_functions:
                raise ValueError("go_prior must match the number of functions.")
            go_prior = go_prior.to(function_init.device, dtype=function_init.dtype)
        return self.function_block(function_init, go_prior)

    def _build_protein_prior(self, pooled: torch.Tensor) -> torch.Tensor:
        """Construct a symmetric, non-negative protein prior from pooled features."""

        if pooled.size(0) == 0:
            return torch.zeros((0, 0), device=pooled.device, dtype=pooled.dtype)

        method = str(self.protein_prior_method).lower()
        if method == "identity":
            return torch.eye(pooled.size(0), device=pooled.device, dtype=pooled.dtype)
        if method != "cosine":
            log.warning("Unknown protein prior method '%s'; defaulting to cosine similarity.", method)

        normed = F.normalize(pooled, dim=1)
        cosine = torch.matmul(normed, normed.T)
        return torch.clamp(cosine, min=0.0)


    def _build_adjacencies(
        self,
        protein_embeddings: torch.Tensor,
        function_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            protein_adj = self.protein_block._adj_from_feats(protein_embeddings)
            function_adj = self.function_block._adj_from_feats(function_embeddings)
        return protein_adj, function_adj
