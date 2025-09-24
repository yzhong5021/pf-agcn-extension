"""
attention.py

self-attention module for PPI and GO relationships + top-K sparsification
"""

from re import L
import torch
import pytorch.nn as nn
import pytorch.nn.functional as F

class GeneralAttention(nn.Module):
    f"""
    general attention mechanism for both PPI and GO relationships. Creates a correlation network between nodes.
    networks are computed as:
    F = V_f * softmax((X_h^(r-1)U_1)U_2(U_3X_h^(r-1)) + b)
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()


        self.W_1 = nn.Parameter(torch.randn(in_dim, out_dim))
        self.W_2 = nn.Parameter(torch.randn(in_dim, out_dim))
        self.W_3 = nn.Parameter(torch.randn(in_dim, out_dim))

        self.V_f = nn.Parameter(torch.randn(out_dim, out_dim))
        self.bias = nn.Parameter(torch.randn(out_dim))
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        L = x @ self.W_1
        R = x @ self.W_3
        
        attn_scores = (L @ self.W_2) @ R.transpose(-2, -1) + self.bias
        
        attn_weights = self.softmax(attn_scores)
        
        output = attn_weights @ self.V_f
        
        return output