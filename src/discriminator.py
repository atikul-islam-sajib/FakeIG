import os
import sys
import torch
import argparse
import torch.nn as nn
from tqdm import tqdm

sys.path.append("./src")

try:
    from patch_embedding import PatchEmbedding
    from transformer_encoder_block import TransformerEncoderBlock
except ImportError:
    print("Import cannot be found".capitalize())


class Discriminator(nn.Module):
    def __init__(
        self,
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
        super(Discriminator, self).__init__()
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

        self.patch_embedding = PatchEmbedding(
            image_size=self.image_size,
            patch_size=self.patch_size,
            image_channels=self.image_channels,
            dimension=self.d_model,
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
                for _ in tqdm(range(self.num_layers))
            ]
        )

        self.classifier = nn.Linear(
            in_features=self.d_model, out_features=self.d_model // self.d_model
        )

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor".capitalize())

        x = self.patch_embedding(x)
        for transformer in self.transformer_blocks:
            x = transformer(x)

        x = torch.mean(input=x, dim=1)
        x = self.classifier(x).view(-1)

        return x


if __name__ == "__main__":
    netD = Discriminator(
        image_size=224,
        patch_size=8,
        num_layers=2,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
        layer_norm_eps=1e-5,
    )
    images = torch.randn((4, 3, 224, 224))
    print(netD(images).shape)
