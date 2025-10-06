# adaptive_function.py
#
# adaptive graphs through bilinear attention module + top-p sparsification
# for function (GO) similarity/hierarchy; UNIDIRECTIONAL graph diffusion
#

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.xavier_uniform_(self.W3.weight)
        for ml in (self.U1, self.U2):
            for lin in ml:
                nn.init.xavier_uniform_(lin.weight)

    def _adj_from_feats(self, X):
        """
        Build adaptive adjacency from node features via bilinear attention.
        X: (N_C, d_in)
        Returns:
            A: (N_C, N_C) row-stochastic attention with fixed p-mass sparsity
        """
        Q, K = self.W1(X), self.W3(X)  # (N_C, d_attn)
        logits = (Q @ self.W2.weight @ K.T) / self.tau  # (N_C, N_C)

        # fixed p-mass cutoff for graph sparsity (per-row)
        vals, idx = torch.sort(logits, dim=1, descending=True)      # sorted rowwise
        probs_sorted = F.softmax(vals, dim=1)                        # convert to probs to cumulate mass
        csum = torch.cumsum(probs_sorted, dim=1)
        k_i = (csum < self.p).sum(dim=1) + 1                         # keep minimum K covering p-mass

        N = logits.size(1)
        ranks = torch.arange(N, device=logits.device).unsqueeze(0).expand_as(vals)
        keep_sorted = ranks < k_i.unsqueeze(1)

        # map mask back to original index order
        neg_inf = vals.new_full(vals.shape, float('-inf'))
        masked_sorted = torch.where(keep_sorted, vals, neg_inf)
        masked = logits.new_full(logits.shape, float('-inf'))
        masked.scatter_(1, idx, masked_sorted)

        A = F.softmax(masked, dim=1)  # row-stochastic after masking
        return A

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
        if S is None:
            S = torch.eye(N, device=device, dtype=dtype)
        else:
            S = S.to(device=device, dtype=dtype)

        I = torch.eye(S.size(0), device=S.device, dtype=dtype)
        S = S + I  # add self loops

        rowsum = S.sum(dim=1, keepdim=True).clamp_min(1e-12)
        R = S / rowsum  # forward-only diffusion

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
