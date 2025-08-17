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
        
        QKV = self.QKV(images)
        query, key, values = torch.chunk(input=QKV, chunks=3, dim=-1)
        
        query = query.view(query.size(0), query.size(1), self.heads, self.dimension // self.heads)
        key = key.view(key.size(0), key.size(1), self.heads, self.dimension // self.heads)
        values = values.view(values.size(0), values.size(1), self.heads, self.dimension // self.heads)
        
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)

        print(query.shape, key.shape, values.shape)
        
        
if __name__ == "__main__":
    attention = MultiHeadAttentionLayer(
        heads=8,
        dimension=512,
        eps=1e-5,
        bias=False
    )
    
    images = torch.randn((16, 64, 512))
    attention(images)