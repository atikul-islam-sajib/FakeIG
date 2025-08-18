import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src")


class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, dimension: int = 512, dropout: float = 0.1, activation: str = "gelu"):
        super(FeedForwardNeuralNetwork, self).__init__()

        self.dimension = dimension
        self.dropout = dropout
        self.activation_func = activation
        
        if activation == "gelu":
            self.activation_func = nn.GELU()
        elif activation == "relu":
            self.activation_func = nn.ReLU()
        elif activation == "swish":
            self.activation_func = nn.SiLU()
        else:
            raise ValueError("Invalid activation function")

        self.layers = []

        for index in range(2):
            if index == 0:
                self.layers.append(
                    nn.Linear(in_features=self.dimension, out_features=4 * self.dimension)
                )
                self.layers.append(self.activation_func)
                self.layers.append(nn.Dropout(p=self.dropout))
            if index != 0:
                self.layers.append(
                    nn.Linear(
                        in_features=4 * self.dimension, out_features=self.dimension
                    )
                )
        self.network = nn.Sequential(*self.layers)
        
    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        
        x = self.network(x)
        return x
        

if __name__ == "__main__":
    network = FeedForwardNeuralNetwork(
        dimension=512,
        dropout=0.1
    )
    print(network)
    images = torch.randn((16, 768, 512))
    print(network(images).size())
