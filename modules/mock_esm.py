"""
mock_esm.py

Stand-in for the ESM1b transformer. Creates initial token embeddings with positional encoding via simple learned embedding + 2-layer MLP.
"""

import torch.nn as nn

N_AA = 21

class MockESM(nn.Module):
    
    def __init__(self, seq_len, hidden_len, embed_len, proj_len):
        super().__init__()
        self.embed = nn.Embedding(N_AA, embed_len)
        self.layer1 = nn.Linear(embed_len, hidden_len)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_len, proj_len)
    
    def forward(self, seqs):
        x = self.embed(seqs)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x