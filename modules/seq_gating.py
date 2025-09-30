"""
seq_gating.py

gating mechanism for esm + dccn embeddings (capture both local and global sequence context)
"""

import torch
import torch.nn as nn

class SeqGating(nn.Module):
    def __init__(self, d_shared = 128, d_esm=1280, c_dcc=256, attn_hidden=128, dropout=0.1):
        super().__init__()

        #project embeddings to d_shared
        self.proj_esm = nn.Linear(d_esm, d_shared, bias=False)
        self.proj_dcc = nn.Linear(c_dcc, d_shared, bias=False)

        # generate token-wise gating weights
        self.gate_tok = nn.Linear(2*d_shared, d_shared)

        # score tokens for attention-based pooling
        self.scorer   = nn.Sequential(nn.Linear(d_shared, attn_hidden), nn.ReLU(), nn.Linear(attn_hidden, 1))

        self.dropout  = nn.Dropout(dropout)
        self.ln       = nn.LayerNorm(d_shared)

    def forward(self, H_esm, H_dcc, lengths):  # (N,L,1280),(N,L,C_dcc),(N,)
        N, L, _ = H_esm.shape # N = batch size, L = max sequence length
        mask = (torch.arange(L, device=H_esm.device)[None,:] < lengths[:,None]).unsqueeze(-1)  # masking sequence length

        # project esm and dcc embeddings to shared d_shared dimension
        E = self.dropout(self.proj_esm(H_esm))
        D = self.dropout(self.proj_dcc(H_dcc))   # both (N,L,d_shared)

        G = torch.sigmoid(self.gate_tok(torch.cat([E, D], dim=-1)))  # (N,L,d_shared) ; generate token-wise gating weights
        H = G*E + (1.0-G)*D                                          # (N,L,d_shared) ; fuse esm and dcc embeddings

        scores = self.scorer(H)                                      # (N,L,1) ; score each token for attention-based pooling
        scores = scores.masked_fill(~mask, float('-inf'))
        w = torch.softmax(scores, dim=1)                             # normalize
        w = torch.nan_to_num(w, nan=0.0)
        x = (H * w).sum(dim=1)                                       # (N,d_shared) ; pooled embedding
        return self.ln(x)                                            # (N,d_shared) ; normalized via layernorm