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
                for _ in tqdm(
                    range(self.num_layers),
                    desc=f"Initializing Transformer with {self.num_layers} layers".capitalize(),
                )
            ]
        )

        self.classifier = nn.Linear(
            in_features=self.d_model,
            out_features=self.d_model // self.d_model,
            bias=False,
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
    parser = argparse.ArgumentParser(description="Discriminator for the FakeIG".title())
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
        "--d_model", type=int, default=512, help="Dimension".capitalize()
    )
    parser.add_argument(
        "--nhead", type=int, default=8, help="Number of heads".capitalize()
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=2048,
        help="Dimension".capitalize(),
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout".capitalize()
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="gelu",
        help="Activation function".capitalize(),
    )
    parser.add_argument(
        "--layer_norm_eps", type=float, default=1e-5, help="Epsilon".capitalize()
    )
    parser.add_argument(
        "--num_layers", type=int, default=4, help="Number of layers".capitalize()
    )

    args = parser.parse_args()

    image_size = args.image_size
    patch_size = args.patch_size
    image_channels = args.image_channels
    d_model = args.d_model
    num_layers = args.num_layers
    nhead = args.nhead
    dim_feedforward = args.dim_feedforward
    dropout = args.dropout
    activation = args.activation
    layer_norm_eps = args.layer_norm_eps

    batch_size = 4

    netD = Discriminator(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        layer_norm_eps=layer_norm_eps,
    )

    images = torch.randn((batch_size, image_channels, image_size, image_size))

    assert (netD(images).shape) == torch.Size(
        [batch_size]
    ), "Output shape is not correct".title()
