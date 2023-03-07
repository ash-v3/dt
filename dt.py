import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecisionTransformer(nn.Module):
    def __init__(self, in_dim, n_heads=8, num_layers=8):
        super().__init__()
        self.in_dim = in_dim
        self.n_heads = n_heads
        self.num_layers = num_layers

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=in_dim, nhead=n_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = F.avg_pool1d(x.t(), kernel_size=x.shape[0]).t()
        return x
