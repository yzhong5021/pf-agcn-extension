"""
adaptive_protein.py

adaptive graphs through bilinear attention module + top-K sparsification for protein similarity; graph diffusion

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveProteinBlock(nn.Module):
    def __init__(self, d_in, d_attn=64, steps=2, p=0.9, tau=1.0, dropout=0.1):
        super().__init__()
        # adaptive graph attention
        self.W1 = nn.Linear(d_in, d_attn, bias=False)
        self.W2 = nn.Linear(d_attn, d_attn, bias=False)
        self.W3 = nn.Linear(d_in, d_attn, bias=False)

        # per-step diffusion weights
        self.U1 = nn.ModuleList([nn.Linear(d_in, d_in, bias = False) for _ in range(steps)]) # forward (prior)
        self.U2 = nn.ModuleList([nn.Linear(d_in, d_in, bias = False) for _ in range(steps)]) # backward (prior)
        self.U3 = nn.ModuleList([nn.Linear(d_in, d_in, bias = False) for _ in range(steps)]) # adaptive

        # graph diffusion (bidirectional)
        self.steps, self.p, self.tau = steps, p, tau
        self.norm = nn.LayerNorm(d_in)

        self.do = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.xavier_uniform_(self.W3.weight)

        for ml in (self.U1, self.U2, self.U3):
            for lin in ml:
                nn.init.xavier_uniform_(lin.weight)

    def _adj_from_feats(self, X):
        Q, K = self.W1(X), self.W3(X)   # (N_P,d_attn)
        logits = (Q @ self.W2.weight @ K.T) / self.tau   # (N_P,N_P) scores ; bilinear transform

        # fixed p-mass cutoff for graph sparsity
        vals, idx = torch.sort(logits, dim=1, descending=True)
        probs_sorted = F.softmax(vals, dim=1)
        csum = torch.cumsum(probs_sorted, dim=1)
        k_i = (csum < self.p).sum(dim=1) + 1

        N = logits.size(1)
        ranks = torch.arange(N, device=logits.device).unsqueeze(0).expand_as(vals)
        keep_sorted = ranks < k_i.unsqueeze(1)

        # convert logit mask to original index space
        masked_sorted = torch.where(keep_sorted, vals, vals.new_full(vals.shape, float('-inf')))
        masked = logits.new_full(logits.shape, float('-inf'))
        masked.scatter_(1, idx, masked_sorted)

        P = F.softmax(masked, dim=1)

        return P

    def forward(self, X, S = None):   # X: (N_P,d_in); S: (N_P,N_P) (optional similarity prior)
        Z = torch.zeros_like(X) # accumulator

        #construct row-stochastic forward/backward prior matrices

        N, device, dtype = X.size(0), X.device, X.dtype
        if S is None:
            S = torch.eye(N, device=device, dtype=dtype)
        else:
            S = S.to(device=device, dtype=dtype)

        I = torch.eye(S.size(0), device=S.device, dtype = dtype)   # self loops

        S = S + I
        rowsum = S.sum(dim=1, keepdim=True).clamp_min(1e-12)
        Rf = S / rowsum
        colsum = S.sum(dim=0, keepdim=True).clamp_min(1e-12)
        Rb = (S.T / colsum).T

        Xf = X.clone()
        Xb = X.clone()
        Xa = X.clone()

        for n in range(self.steps):
            A = self._adj_from_feats(Xa) # (N_P,N_P); adaptive graph

            # prior mixing; adaptive features
            Xf = Rf @ Xf
            Xb = Rb @ Xb
            Xa = A @ Xa

            Pf = self.U1[n](Xf) # (N_P,d_in)
            Pb = self.U2[n](Xb) # (N_P,d_in)
            P = Pf + Pb
        
            # adaptive forward
            F = self.U3[n](Xa)

            delta = self.do(P+F)

            Z = Z + delta # (N_P,d_in)
        # residual gate
        return self.norm(X + Z) # (N_P,d_in)
