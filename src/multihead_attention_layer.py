import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

try:
    from scale_dot_product import scale_dot_product
except ImportError:
    print("scale_dot_product.py not found".capitalize())


class MultiHeadAttentionLayer(nn.Module):
    def __init__(
        self,
        heads: int = 8,
        dimension: int = 512,
        eps: float = 1e-5,
        bias: bool = False,
    ):
        super(MultiHeadAttentionLayer, self).__init__()
        self.heads = heads
        self.dimension = dimension
        self.eps = eps
        self.bias = bias

        assert (
            self.dimension % self.heads == 0
        ), "Dimension must be divisible by heads".capitalize()

        self.QKV = nn.Linear(
            in_features=self.dimension, out_features=self.dimension * 3, bias=False
        )

    def forward(self, images: torch.Tensor):
        if not isinstance(images, torch.Tensor):
            raise TypeError("Images must be a torch.Tensor".capitalize())

        QKV = self.QKV(images)
        query, key, values = torch.chunk(input=QKV, chunks=3, dim=-1)

        query = query.view(
            query.size(0), query.size(1), self.heads, self.dimension // self.heads
        )
        key = key.view(
            key.size(0), key.size(1), self.heads, self.dimension // self.heads
        )
        values = values.view(
            values.size(0), values.size(1), self.heads, self.dimension // self.heads
        )

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)

        attention = scale_dot_product(query=query, key=key, value=values)

        attention = attention.permute(0, 2, 1, 3)
        attention = attention.reshape(
            attention.size(0),
            attention.size(1),
            attention.size(2) * attention.size(3),
        )

        return attention


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MultiHead Attention Layer".title())
    parser.add_argument(
        "--heads", type=int, default=8, help="Number of heads".capitalize()
    )
    parser.add_argument(
        "--dimension", type=int, default=512, help="Dimension".capitalize()
    )
    parser.add_argument("--eps", type=float, default=1e-5, help="Epsilon".capitalize())
    parser.add_argument("--bias", type=bool, default=False, help="Bias".capitalize())

    args = parser.parse_args()

    attention = MultiHeadAttentionLayer(
        heads=args.heads, dimension=args.dimension, eps=args.eps, bias=args.bias
    )

    images = torch.randn((16, 64, args.dimension))

    assert (
        attention(images).size()
    ) == images.size(), "Attention must be the same size as images".capitalize()
