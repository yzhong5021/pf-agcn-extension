"""
dccn.py

1-D dilated causal convolutional network (WaveNet). Called after sequences are embedded by ESM-1b.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class DCCN_1D(nn.Module):
    
    def __init__(self, embed_len):
        super().__init__()
        k = 2
        self.dilation = (1, 2, 4, 8)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_len, out_channels=embed_len, kernel_size=k, dilation=d, padding=0)
            for d in self.dilation
        ])

        self.gating = nn.Conv1d(in_channels = 4 * embed_len, out_channels = 4, kernel_size = 1) #dilation channelwise linear gating

    def forward(self, x):
        # x = B, L, C where B = batch size, L = sequence length, C = embedding length
        xt = x.transpose(1,2) # make x B, C, L

        qs = []

        for dil, conv in zip(self.dilation, self.convs):
            x_mask = F.pad(xt, (dil, 0), value = 0) # causal masking via padding on L
            qs.append(F.relu(conv(x_mask))) #apply convolution
        
        # gating
        q_concat = torch.cat(qs, dim = 1)

        alpha = torch.sigmoid(self.gating(q_concat)) # a = B, 4, L
        alphas = torch.unbind(alpha, dim=1)

        out = 0

        for a, q in zip(alphas, qs):
            a = a.unsqueeze(1) # allow for dilation channel
            out = out + a * q # accumulate gated features

        return out.transpose(1, 2) # revert back to B, L, C


