# adaptive_function.py
#
# adaptive graphs through bilinear attention module + top-p sparsification
# for function (GO) similarity/hierarchy; UNIDIRECTIONAL graph diffusion
#

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.adaptive_helpers import (
    _init_attention_layers,
    _init_linear_stack,
    _build_top_p_attention,
    _prepare_prior,
)

class AdaptiveFunctionBlock(nn.Module):
    def __init__(self, d_in, d_attn=64, steps=2, p=0.9, tau=1.0, dropout=0.1):
        super().__init__()
        # adaptive graph attention (bilinear)
        self.W1 = nn.Linear(d_in, d_attn, bias=False)
        self.W2 = nn.Linear(d_attn, d_attn, bias=False)
        self.W3 = nn.Linear(d_in, d_attn, bias=False)

        # per-step diffusion weights
        # unidirectional diffusion
        self.U1 = nn.ModuleList([nn.Linear(d_in, d_in, bias=False) for _ in range(steps)])  # forward (prior)
        self.U2 = nn.ModuleList([nn.Linear(d_in, d_in, bias=False) for _ in range(steps)])  # adaptive

        # graph diffusion (UNIDIRECTIONAL)
        self.steps, self.p, self.tau = steps, p, tau
        self.norm = nn.LayerNorm(d_in)
        self.do = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        _init_attention_layers(self.W1, self.W2, self.W3)
        for ml in (self.U1, self.U2):
            _init_linear_stack(ml)

    def _adj_from_feats(self, X):
        """
        Build adaptive adjacency from node features via bilinear attention.
        X: (N_C, d_in)
        Returns:
            A: (N_C, N_C) row-stochastic attention with fixed p-mass sparsity
        """
        return _build_top_p_attention(self.W1, self.W2, self.W3, X, self.tau, self.p)

    def forward(self, X, S=None):
        """
        X: (N_C, d_in)         function (class) embeddings/features
        S: (N_C, N_C) optional directed prior adjacency (e.g., GO DAG).
           If None, defaults to identity (self-loops only).
        Returns:
            (N_C, d_in)
        """

        # accumulator
        Z = torch.zeros_like(X)

        # construct row-stochastic forward prior matrix (with self-loops)
        N, device, dtype = X.size(0), X.device, X.dtype
        R = _prepare_prior(S, N, device, dtype, bidirectional=False)

        # states
        Xf = X.clone()  # prior-forward state
        Xa = X.clone()  # adaptive state

        for n in range(self.steps):
            # adaptive graph from features (recomputed every step)
            A = self._adj_from_feats(Xa)     # (N_C, N_C)

            # unidirectional prior mixing + adaptive features
            Xf = R @ Xf        # (N_C, d_in)
            Xa = A @ Xa        # (N_C, d_in)

            Pf = self.U1[n](Xf)   # prior contribution
            F_ = self.U2[n](Xa)   # adaptive contribution

            delta = self.do(Pf + F_)
            Z = Z + delta

        # residual + layernorm
        return self.norm(X + Z)
