import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src")

try:
    from patch_embedding import PatchEmbedding
except ImportError:
    print("Import cannot be found".capitalize())
    

class Discriminator(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 8,
        num_layers: int = 4,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
    ):
        super(Discriminator, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        
    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor".capitalize())
        
        
if __name__ == "__main__":
    netD = Discriminator(
        image_size=224,
        patch_size=8,
        num_layers=4,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,    
        activation="gelu",
        layer_norm_eps=1e-5
    )
