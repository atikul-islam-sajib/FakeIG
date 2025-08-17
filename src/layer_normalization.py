import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src")

class LayerNormalization(nn.Module):
    def __init__(self, dimension: int = 512, eps: float = 1e-5):
        super(LayerNormalization, self).__init__()
        
        self.dimension = dimension
        self.eps = eps
        
        self.alpha = nn.Parameter(torch.ones((self.dimension//self.dimension, self.dimension//self.dimension, self.dimension)))
        self.gamma = nn.Parameter(torch.zeros((self.dimension//self.dimension, self.dimension//self.dimension, self.dimension)))
        
    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor".capitalize())
        
if __name__ == "__main__":
    normalization = LayerNormalization(dimension=512, eps=1e-5)
    
    images = torch.randn((16, 64, 512))