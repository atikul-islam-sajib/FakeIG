import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src")


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
    ):
        super(TransformerEncoderBlock, self).__init__()
        self.d_model = d_model
        self.nheads = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        
    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor".capitalize())
        
if __name__ == "__main__":
    transformer = TransformerEncoderBlock(
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
        layer_norm_eps=1e-5
    )
