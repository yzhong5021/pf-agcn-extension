"""
adaptive_protein.py

adaptive graphs through bilinear attention module + top-K sparsification for protein similarity; graph diffusion

"""

import torch
import torch.nn as nn

from src.utils.adaptive_helpers import (
    _init_attention_layers,
    _init_linear_stack,
    _build_top_p_attention,
    _prepare_prior,
)


class AdaptiveProteinBlock(nn.Module):
    def __init__(self, d_in, d_attn=64, steps=2, p=0.9, tau=1.0, dropout=0.1):
        super().__init__()
        # adaptive graph attention
        self.W1 = nn.Linear(d_in, d_attn, bias=False)
        self.W2 = nn.Linear(d_attn, d_attn, bias=False)
        self.W3 = nn.Linear(d_in, d_attn, bias=False)

        # per-step diffusion weights
        self.U1 = nn.ModuleList([nn.Linear(d_in, d_in, bias=False) for _ in range(steps)])  # forward (prior)
        self.U2 = nn.ModuleList([nn.Linear(d_in, d_in, bias=False) for _ in range(steps)])  # backward (prior)
        self.U3 = nn.ModuleList([nn.Linear(d_in, d_in, bias=False) for _ in range(steps)])  # adaptive

        # graph diffusion (bidirectional)
        self.steps, self.p, self.tau = steps, p, tau
        self.norm = nn.LayerNorm(d_in)

        self.do = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        _init_attention_layers(self.W1, self.W2, self.W3)
        for ml in (self.U1, self.U2, self.U3):
            _init_linear_stack(ml)

    def _adj_from_feats(self, X):
        return _build_top_p_attention(self.W1, self.W2, self.W3, X, self.tau, self.p)

    def forward(self, X, S=None):  # X: (N_P,d_in); S: (N_P,N_P) (optional similarity prior)
        Z = torch.zeros_like(X)  # accumulator

        # construct row-stochastic forward/backward prior matrices
        N, device, dtype = X.size(0), X.device, X.dtype
        Rf, Rb = _prepare_prior(S, N, device, dtype, bidirectional=True)

        Xf = X.clone()
        Xb = X.clone()
        Xa = X.clone()

        for n in range(self.steps):
            A = self._adj_from_feats(Xa)  # (N_P,N_P); adaptive graph

            # prior mixing; adaptive features
            Xf = Rf @ Xf
            Xb = Rb @ Xb
            Xa = A @ Xa

            Pf = self.U1[n](Xf)  # (N_P,d_in)
            Pb = self.U2[n](Xb)  # (N_P,d_in)
            P = Pf + Pb

            # adaptive forward
            F = self.U3[n](Xa)

            delta = self.do(P + F)

            Z = Z + delta  # (N_P,d_in)
        # residual gate
        return self.norm(X + Z)  # (N_P,d_in)
