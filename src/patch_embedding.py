import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src")

class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int = 224, patch_size: int = 8):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        
    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input should be in torch format".capitalize())
        
if __name__ == "__main__":
    embedding = PatchEmbedding(
        image_size=224,
        patch_size=8
    )