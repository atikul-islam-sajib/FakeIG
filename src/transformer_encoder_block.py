import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src")

try:
    from layer_normalization import LayerNormalization
    from multihead_attention_layer import MultiHeadAttentionLayer
    from feedforward_neural_network import FeedForwardNeuralNetwork
except ImportError:
    print("Import cannot be found".capitalize())


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
    ):
        super(TransformerEncoderBlock, self).__init__()
        self.d_model = d_model
        self.nheads = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps

        self.normalization1 = LayerNormalization(
            dimension=self.d_model, eps=self.layer_norm_eps
        )
        self.normalization2 = LayerNormalization(
            dimension=self.d_model, eps=self.layer_norm_eps
        )

        self.dropout1 = nn.Dropout(p=self.dropout)
        self.dropout2 = nn.Dropout(p=self.dropout)

        self.multihead_attention = MultiHeadAttentionLayer(
            heads=self.nheads,
            dimension=self.d_model,
            eps=self.layer_norm_eps,
        )
        self.neural_network = FeedForwardNeuralNetwork(
            dimension=self.d_model,
            dropout=self.dropout,
            activation=self.activation,
        )

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor".capitalize())

        residual = x

        x = self.multihead_attention(x)
        x = self.dropout1(x)
        x = torch.add(input=x, other=residual)
        x = self.normalization1(x)

        residual = x

        x = self.neural_network(x)
        x = self.dropout2(x)
        x = torch.add(input=x, other=residual)
        x = self.normalization2(x)

        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer Encoder Block".title())
    parser.add_argument(
        "--d_model", type=int, default=512, help="Dimension".capitalize()
    )
    parser.add_argument(
        "--nhead", type=int, default=8, help="Number of heads".capitalize()
    )
    parser.add_argument(
        "--dim_feedforward", type=int, default=2048, help="Dimension".capitalize()
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

    args = parser.parse_args()

    d_model = args.d_model
    nhead = args.nhead
    dim_feedforward = args.dim_feedforward
    dropout = args.dropout
    activation = args.activation
    layer_norm_eps = args.layer_norm_eps

    transformer = TransformerEncoderBlock(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        layer_norm_eps=layer_norm_eps,
    )

    images = torch.randn((16, 768, d_model))
    assert (
        transformer(images).size()
    ) == images.size(), "Transformer must be the same size as images".capitalize()
