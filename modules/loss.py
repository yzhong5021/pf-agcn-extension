"""
loss.py

evaluate + BCE with logits loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEWithLogits(nn.Module):

    def __init__(self, pos_weight):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits, targets):

        loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )

        return loss.mean()
