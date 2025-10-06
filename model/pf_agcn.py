"""PF-AGCN end-to-end pipeline.

Stitches sequence encoders, adaptive graph modules, and classifier for protein
function prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from modules.dccn import DCCN_1D
from modules.seq_gating import SeqGating
from modules.seq_final import SeqFinal
from modules.adaptive_protein import AdaptiveProteinBlock
from modules.adaptive_function import AdaptiveFunctionBlock
from modules.head import ClassificationHead


@dataclass
class PFAGCNOutput:
    """Container for PF-AGCN forward pass artifacts."""

    logits: torch.Tensor
    protein_embeddings: torch.Tensor
    function_embeddings: torch.Tensor
    protein_adjacency: torch.Tensor
    function_adjacency: torch.Tensor


class PFAGCN(nn.Module):
    """Protein Function Adaptive Graph Convolutional Network.

    Args:
        num_functions: Number of function classes (GO terms).
        seq_encoder: Sequence encoder producing contextualized amino acid
            embeddings.
        seq_encoder_dim: Feature width yielded by ``seq_encoder``.
        shared_dim: Width of the shared token space used in sequence gating.
        graph_dim: Node embedding width for graph refinement and classification.
        metric_dim: Metric projection width used for feature matching.
        dccn_channels: Channel count for the dilated CNN block.
        dccn_kernel: Kernel size for the dilated CNN layers.
        dccn_dilation: Base dilation factor for the dilated CNN stack.
        dropout: Dropout probability reused across submodules.
        protein_steps: Diffusion steps for the protein adaptive block.
        function_steps: Diffusion steps for the function adaptive block.
        graph_keep_mass: Cumulative probability mass retained per node when
            sparsifying adaptive graphs.
        graph_tau: Temperature applied to adaptive attention logits.
    """

    def __init__(
        self,
        num_functions: int,
        seq_encoder: nn.Module,
        seq_encoder_dim: int,
        *,
        shared_dim: int = 128,
        graph_dim: int = 128,
        metric_dim: int = 64,
        dccn_channels: int = 64,
        dccn_kernel: int = 3,
        dccn_dilation: int = 2,
        dropout: float = 0.1,
        protein_steps: int = 2,
        function_steps: int = 2,
        graph_keep_mass: float = 0.9,
        graph_tau: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_functions = num_functions
        self.seq_encoder = seq_encoder
        self.seq_encoder_dim = seq_encoder_dim

        if seq_encoder_dim != dccn_channels:
            self.dccn_input = nn.Linear(
                seq_encoder_dim, dccn_channels, bias=False
            )
            nn.init.xavier_uniform_(self.dccn_input.weight)
        else:
            self.dccn_input = nn.Identity()

        self.dccn = DCCN_1D(
            embed_len=dccn_channels,
            k_size=dccn_kernel,
            dilation=dccn_dilation,
            dropout=dropout,
        )
        self.seq_gate = SeqGating(
            d_shared=shared_dim,
            d_esm=seq_encoder_dim,
            c_dcc=dccn_channels,
            attn_hidden=shared_dim,
            dropout=dropout,
        )
        self.seq_final = SeqFinal(
            in_dim=shared_dim,
            N_C=num_functions,
            proj=metric_dim,
            out_ch=graph_dim,
        )
        self.protein_block = AdaptiveProteinBlock(
            d_in=graph_dim,
            d_attn=graph_dim,
            steps=protein_steps,
            p=graph_keep_mass,
            tau=graph_tau,
            dropout=dropout,
        )
        self.function_block = AdaptiveFunctionBlock(
            d_in=graph_dim,
            d_attn=graph_dim,
            steps=function_steps,
            p=graph_keep_mass,
            tau=graph_tau,
            dropout=dropout,
        )
        self.head = ClassificationHead(
            N_C=num_functions,
            d_in=graph_dim,
            dropout=dropout,
        )

    def forward(
        self,
        seq_tokens: torch.Tensor,
        seq_lengths: torch.Tensor,
        protein_prior: Optional[torch.Tensor] = None,
        function_prior: Optional[torch.Tensor] = None,
    ) -> PFAGCNOutput:
        """Run the PF-AGCN inference pipeline.

        Args:
            seq_tokens: Integer-encoded protein sequences of shape ``(N, L)``.
            seq_lengths: Valid sequence lengths for each protein,
                shaped ``(N,)``.
            protein_prior: Optional protein adjacency prior of shape ``(N, N)``.
            function_prior: Optional function adjacency prior of shape
                ``(C, C)``.

        Returns:
            PFAGCNOutput: Model logits and intermediate graph features.
        """
        if seq_tokens.ndim != 2:
            raise ValueError("seq_tokens must be a 2D tensor (batch, length).")

        if seq_lengths.ndim != 1:
            raise ValueError("seq_lengths must be a 1D tensor of lengths.")

        device = next(self.parameters()).device
        seq_tokens = seq_tokens.to(device=device)
        seq_lengths = seq_lengths.to(device=device, dtype=torch.long)

        esm_embeddings = self.seq_encoder(seq_tokens)
        conv_input = self.dccn_input(esm_embeddings)
        conv_features = self.dccn(conv_input)
        protein_repr = self.seq_gate(esm_embeddings, conv_features, seq_lengths)

        protein_init, function_init = self._initial_graph_features(protein_repr)

        if protein_prior is not None:
            if protein_prior.ndim != 2:
                raise ValueError("protein_prior must be 2D (N, N).")
            n_proteins = protein_init.size(0)
            if (
                protein_prior.shape[0] != n_proteins
                or protein_prior.shape[1] != n_proteins
            ):
                raise ValueError(
                    "protein_prior must match the protein batch size."
                )
            protein_prior = protein_prior.to(
                device=protein_init.device, dtype=protein_init.dtype
            )

        if function_prior is not None:
            if function_prior.ndim != 2:
                raise ValueError("function_prior must be 2D (C, C).")
            if (
                function_prior.shape[0] != self.num_functions
                or function_prior.shape[1] != self.num_functions
            ):
                raise ValueError(
                    "function_prior must match the number of functions."
                )
            function_prior = function_prior.to(
                device=function_init.device, dtype=function_init.dtype
            )

        protein_embeddings = self.protein_block(protein_init, protein_prior)
        function_embeddings = self.function_block(
            function_init, function_prior
        )

        logits = self.head(function_embeddings, protein_embeddings)

        with torch.no_grad():
            protein_adjacency = self.protein_block._adj_from_feats(
                protein_embeddings
            )
            function_adjacency = self.function_block._adj_from_feats(
                function_embeddings
            )

        return PFAGCNOutput(
            logits=logits,
            protein_embeddings=protein_embeddings,
            function_embeddings=function_embeddings,
            protein_adjacency=protein_adjacency,
            function_adjacency=function_adjacency,
        )

    def _initial_graph_features(
        self, protein_repr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build initial protein and function node states.

        Args:
            protein_repr: Sequence-derived protein descriptors of shape
                ``(N, shared_dim)``.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Protein features shaped
            ``(N, graph_dim)`` and function features shaped ``(C, graph_dim)``.
        """
        metric = self.seq_final.met_proj(protein_repr)
        a = metric.unsqueeze(1).expand(-1, metric.size(0), -1)
        b = metric.unsqueeze(0).expand(metric.size(0), -1, -1)

        comp = torch.cat(
            (a, b, torch.abs(a - b), a * b),
            dim=-1,
        )
        pairwise = self.seq_final.norm(self.seq_final.mlp(comp))
        protein_features = torch.diagonal(pairwise, dim1=0, dim2=1).contiguous()

        function_features = self.seq_final.go_proj(protein_repr).mean(dim=1)

        return protein_features, function_features
