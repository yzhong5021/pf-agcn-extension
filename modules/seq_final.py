"""
seq_final.py

transforms sequence embeddings from seq_gating.py into graph-ready features. used to initialize both 
protein similarity and GO embeddings.
"""

import torch
import torch.nn as nn

class SeqFinal(nn.Module):
    def __init__(self, in_dim, N_C, proj=64, out_ch=64):
        super().__init__()
        self.met_proj  = nn.Linear(in_dim, proj, bias=True) # project to metric space

        self.G = nn.Parameter(torch.empty(N_C, proj))

        comp_in = 4*proj
        self.mlp = nn.Sequential(
            nn.Linear(comp_in, 2*comp_in),
            nn.GELU(),
            nn.Linear(2*comp_in, out_ch),
        )
        self.norm = nn.LayerNorm(out_ch)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.met_proj.weight)
        nn.init.zeros_(self.met_proj.bias)

        nn.init.xavier_uniform_(self.G)

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def prot_proj(self, x):  # protein-protein: (N_P, d) -> (N_P, N_P, C)
        a = self.met_proj(x)                  # (N_P, proj)
        A = a[:, None, :].expand(a.size(0), a.size(0), -1)                      # (N_P, N_P, proj)
        B = a[None, :, :].expand(a.size(0), a.size(0), -1)                      # (N_P, N_P, proj)

        print(A.shape, B.shape)
        comp = torch.cat([A, B, (A-B).abs(), A*B], dim=-1)  # (N_P,N_P,4*proj); learn simple distance/agreement metrics

        return self.norm(self.mlp(comp)) # (N_P, N_P, C)

    def go_proj(self, x): # GO-protein: (N_P, d) -> (N_C, N_P, C)
        a = self.met_proj(x)                  # (N_P, proj)

        G_ = self.G.unsqueeze(1).expand(-1, a.size(0), -1) # (N_C, N_P, proj)
        P_ = a.unsqueeze(0).expand(self.G.size(0), -1, -1) # (N_C, N_P, proj)

        comp = torch.cat([P_, G_, (P_-G_).abs(), P_*G_], dim=-1)  # (N_C,N_P,4*proj)

        return self.norm(self.mlp(comp)) # (N_C, N_P, C)