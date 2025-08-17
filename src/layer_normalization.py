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

        self.gamma = nn.Parameter(
            torch.ones(
                (
                    self.dimension // self.dimension,
                    self.dimension // self.dimension,
                    self.dimension,
                )
            )
        )
        self.betas = nn.Parameter(
            torch.zeros(
                (
                    self.dimension // self.dimension,
                    self.dimension // self.dimension,
                    self.dimension,
                )
            )
        )

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor".capitalize())

        mean = torch.mean(input=x, dim=-1, keepdim=True)
        variance = torch.var(input=x, dim=-1, keepdim=True)

        x = (x - mean) / torch.sqrt(variance + self.eps)

        return self.gamma * x + self.betas


if __name__ == "__main__":
    normalization = LayerNormalization(dimension=512, eps=1e-5)

    images = torch.randn((16, 64, 512))

    print(normalization(images).size())
