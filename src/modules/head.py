"""
head.py

feature fusion + classification head
"""

import torch
import torch.nn as nn

class _Proj(nn.Module):
    def __init__(self, d_in, p_drop=0.1):
        super().__init__()
        self.l1 = nn.Linear(d_in, d_in)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(p_drop)
        self.l2 = nn.Linear(d_in, d_in)
        self.norm = nn.LayerNorm(d_in)

    def forward(self, x):
        y = self.norm(self.l2(self.drop(self.gelu(self.l1(x)))) + x)

        return y

class ClassificationHead(nn.Module):
    def __init__(self, N_C, d_in, dropout=0.1):
        super().__init__()

        self.p_proj = _Proj(d_in, p_drop = dropout)
        self.f_proj = _Proj(d_in, p_drop = dropout)

        self.bias = nn.Parameter(torch.zeros(N_C))
        self.log_tau = nn.Parameter(torch.zeros(())) # force positive

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.log_tau)

    def forward(self, F, P): # F: (N_C, d_in), P: (N_P, d_in)

        P = self.p_proj(P)
        F = self.f_proj(F)

        logits = torch.exp(self.log_tau) * (P @ F.T) / P.size(-1)**0.5 + self.bias # (N_P, N_C)

        return logits
