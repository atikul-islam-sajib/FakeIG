import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src")


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 8,
        image_channels: int = 3,
        dimension: int = 512,
    ):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.image_channels = image_channels
        self.dimension = dimension

        assert (
            self.image_size % self.patch_size == 0
        ), "Image size must be divisible by patch size".title()

        self.num_of_patches = (self.image_size // self.patch_size) ** 2

        self.kernel_size = patch_size
        self.stride_size = patch_size
        self.padding_size = int(self.patch_size // self.patch_size)

        if self.dimension == 0:
            self.dimension = (self.padding_size**2) * self.image_channels

        self.projection = nn.Conv2d(
            in_channels=self.image_channels,
            out_channels=self.dimension,
            kernel_size=self.kernel_size,
            stride=self.stride_size,
            padding=self.padding_size,
        )

        self.positional_encoding = nn.Parameter(
            torch.randn(
                self.patch_size // self.patch_size,
                self.patch_size // self.patch_size,
                self.dimension,
            ),
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input should be in torch format".capitalize())

        x = self.projection(x)
        x = x.view(x.size(0), x.size(1), x.size(2) * x.size(3))
        x = x.permute(0, 2, 1)
        x = x + self.positional_encoding

        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch Embedding".title())
    parser.add_argument(
        "--image_size", type=int, default=224, help="Image size".capitalize()
    )
    parser.add_argument(
        "--patch_size", type=int, default=8, help="Patch size".capitalize()
    )
    parser.add_argument(
        "--image_channels", type=int, default=3, help="Image channels".capitalize()
    )
    parser.add_argument(
        "--dimension", type=int, default=512, help="Dimension".capitalize()
    )

    args = parser.parse_args()

    embedding = PatchEmbedding(
        image_size=args.image_size,
        patch_size=args.patch_size,
        image_channels=args.image_channels,
        dimension=args.dimension,
    )
    images = torch.randn((16, 3, args.image_size, args.image_size))
    print(embedding(images).size())
