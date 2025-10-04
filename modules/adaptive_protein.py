"""
attention.py

adaptive graphs through bilinear attention module + top-K sparsification for protein similarity; graph diffusion

"""

from re import L
import torch
import pytorch.nn as nn
import pytorch.nn.functional as F

class AdaptiveProteinBlock(nn.Module):
    def __init__(self, d_in, d_attn=64, steps=2, topk=16, tau=1.0,
                 bidir=False, dropout=0.1):
        super().__init__()
        # edge scorer (bilinear attention)
        self.W1 = nn.Linear(d_in, d_attn, bias=False)
        self.W2 = nn.Linear(d_attn, d_attn, bias=False)
        self.W3 = nn.Linear(d_in, d_attn, bias=False)
        # per-hop mixers
        self.mix = nn.ModuleList(nn.Linear(d_in, d_in) for _ in range(steps))
        self.steps, self.topk, self.tau, self.bidir = steps, topk, tau, bidir
        self.do = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_in)

    def _adj_from_feats(self, X):
        Q, K = self.W1(X), self.W3(X)                     # (N,d_attn)
        S = (Q @ self.W2.weight @ K.T) / self.tau         # (N,N) scores
        # Top-K per row (indices non-diff, scores diff)
        idx = torch.topk(S, self.topk, dim=1).indices
        mask = S.new_full(S.shape, float('-inf')); mask.scatter_(1, idx, 0.0)
        P = torch.softmax(S + mask, dim=1)                # row-stochastic \tilde P
        return self.do(P)

    def forward(self, X):                                  # X: (N,d_in)
        P = self._adj_from_feats(X)
        H, Z = X, 0
        for n in range(self.steps):
            Hf = P @ H
            Hb = P.T @ H if self.bidir else 0
            H = Hf + Hb
            Z = Z + self.mix[n](H)
        # residual + norm
        return self.norm(X + Z)
