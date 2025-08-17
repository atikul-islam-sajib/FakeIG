import os
import sys
import torch
import torch.nn as nn

sys.path.append("./src/")

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, heads: int = 8, dimension: int = 512, eps: float = 1e-5, bias: bool = False):
        super(MultiHeadAttentionLayer, self).__init__()
        self.heads = heads
        self.dimension = dimension
        self.eps = eps
        self.bias = bias
        
        assert self.dimension % self.heads == 0, "Dimension must be divisible by heads".capitalize()
        
        self.QKV = nn.Linear(in_features=self.dimension, out_features=self.dimension * 3, bias=False)
        
    def forward(self, images: torch.Tensor):
        if not isinstance(images, torch.Tensor):
            raise TypeError("Images must be a torch.Tensor".capitalize())
        
        
if __name__ == "__main__":
    attention = MultiHeadAttentionLayer(
        heads=8,
        dimension=512,
        eps=1e-5,
        bias=False
    )