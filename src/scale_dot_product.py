import os
import sys
import torch
import torch.nn as nn

sys.path.append("./src/")


def scale_dot_product(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
    if (
        not isinstance(query, torch.Tensor)
        or not isinstance(key, torch.Tensor)
        or not isinstance(value, torch.Tensor)
    ):
        raise TypeError("All inputs must be torch.Tensor")


if __name__ == "__main__":
    pass