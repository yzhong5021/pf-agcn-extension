"""
seq_gating.py

gating mechanism for esm + dccn embeddings (capture both local and global sequence context)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPool(nn.Module):
    def __init__(self, K, hidden=128):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(K, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, H, lengths):
        B, L, K = H.shape
        scores = self.scorer(H)                            # (B, L, 1)
        mask = (torch.arange(L, device=H.device)[None,:] < lengths[:,None]).unsqueeze(-1)  # (B,L,1)
        scores = scores.masked_fill(~mask, float('-inf'))
        w = torch.softmax(scores, dim=1)                  # (B, L, 1)
        w = torch.nan_to_num(w, nan=0.0)
        return (H * w).sum(dim=1)    

class SeqGating(nn.Module):
    def __init__(self, embed_len, gating_len):
        super().__init__()
        self.gating = nn.Linear(embed_len, gating_len)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.AdaptiveMaxPool1d(1)
