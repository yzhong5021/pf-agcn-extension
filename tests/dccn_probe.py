"""
dccn_probe.py

Linear probe for testing DCCN module
"""

import torch.nn as nn
import torch.nn.functional as F

class LinearProbe(nn.Module):

    def __init__(self, in_dim, go_dim):
        super().__init__()
        self.probe = nn.Linear(in_dim, go_dim)

    def forward(self, x):
        return self.probe(x)
