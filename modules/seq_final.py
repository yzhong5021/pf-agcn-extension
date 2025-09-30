"""
seq_final.py

transforms sequence embeddings from seq_gating.py into GO-centric features and integrates GO hierarchy.
"""

import torch
import torch.nn as nn

class SeqFinal(nn.Module):
    """
    transforms sequence embeddings from seq_gating.py into GO-centric features and integrates GO hierarchy.
    
    args:
        d_shared (int): dimension of sequence embeddings
        n_classes (int): number of GO classes
        d_go (int): dimension of GO embeddings
        n_p (int): batch size

    returns:
        x (torch.Tensor): GO-centric features
    """

    def __init__(self, d_shared = 128, n_classes = 1957, d_go = 16, p_emb = 32, n_p = 25):
        super().__init__()

        self.d_shared = d_shared
        self.d_go = d_go
        self.n_p = n_p
        self.n_classes = n_classes
        self.p_emb = p_emb

        self.go_expand = nn.Linear(1, self.n_classes)
        self.go_fit = nn.Linear(self.d_shared, self.p_emb)

        self.batch_norm = nn.BatchNorm1d(d_go)

    def forward(self, x, go_embed): # (n_p,d_shared),(n_classes,d_go)
        x = x.unsqueeze(-1) # (n_p, d_shared, 1)
        x = self.go_expand(x) # (n_p, d_shared, n_classes)
        x = x.transpose(-2, -1) # (n_p, n_classes, d_shared)
        x = self.go_fit(x) # (n_p, n_classes, p_emb)

        go_embed = go_embed.unsqueeze(0).expand(self.n_p, -1, -1) # (n_p, n_classes, d_go)

        x = torch.cat([x, go_embed], dim=-1) # (n_p,n_classes,p_emb + d_go)
        x = self.batch_norm(x)
        return x
