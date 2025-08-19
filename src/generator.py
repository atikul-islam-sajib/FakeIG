import os
import sys
import math
import torch
import argparse
import torch.nn as nn
from tqdm import tqdm

sys.path.append("./src")

try:
    from transformer_encoder_block import TransformerEncoderBlock
except ImportError:
    print("Import cannot be found".capitalize())


class Decoder(nn.Module):
    def __init__(
        self, image_size: int = 224, patch_size: int = 8, in_channels: int = 512
    ):
        super(Decoder, self).__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = self.in_channels // 2
        self.kernel_size = 4
        self.stride_size = 2
        self.padding_size = self.kernel_size // self.kernel_size

        self.layers = []

        self.repetative = int(
            math.log2((self.image_size // (self.image_size // self.patch_size)))
        )

        for idx in range(self.repetative):
            self.layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride_size,
                    padding=self.padding_size,
                )
            )
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.BatchNorm2d(self.out_channels))

            self.in_channels = self.out_channels
            self.out_channels = self.in_channels // 2

        self.layers.append(
            nn.Conv2d(in_channels=self.in_channels, out_channels=3, kernel_size=1)
        )
        self.layers.append(nn.Tanh())

        self.network = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor".capitalize())

        return self.network(x)


class Generator(nn.Module):
    def __init__(
        self,
        latent_size: int = 100,
        image_channels: int = 3,
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
        super(Generator, self).__init__()

        self.latent_size = latent_size
        self.image_channels = image_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps

        self.num_of_patches = (self.image_size // self.patch_size) ** 2

        self.projection = nn.Linear(
            in_features=self.latent_size,
            out_features=self.num_of_patches * self.d_model,
        )

        self.positional_encoding = nn.Parameter(
            torch.ones((1, 1, self.d_model)), requires_grad=True
        )

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    dim_feedforward=self.dim_feedforward,
                    dropout=self.dropout,
                    activation=self.activation,
                    layer_norm_eps=self.layer_norm_eps,
                )
                for _ in tqdm(
                    range(self.num_layers),
                    desc=f"Initializing Transformer with {self.num_layers} layers".capitalize(),
                )
            ]
        )

        self.decoder = Decoder(
            image_size=self.image_size,
            patch_size=self.patch_size,
            in_channels=self.d_model
        )

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor".capitalize())

        x = x.squeeze()
        x = self.projection(x)
        x = x.view(x.size(0), self.num_of_patches, self.d_model)

        x = x + self.positional_encoding

        for transformer in self.transformer_blocks:
            x = transformer(x)

        x = x.permute(0, 2, 1)
        x = x.view(
            x.size(0),
            x.size(1),
            (self.image_size // self.patch_size),
            (self.image_size // self.patch_size),
        )

        x = self.decoder(x)
        
        return x


if __name__ == "__main__":
    netG = Generator(
        image_channels=3,
        image_size=224,
        patch_size=8,
        num_layers=1,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
        layer_norm_eps=1e-5,
    )

    images = torch.randn((16, 100, 1, 1))

    print(netG(images).size())
