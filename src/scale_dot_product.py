import os
import sys
import math
import torch
import torch.nn as nn

sys.path.append("./src/")


def scale_dot_product(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
    if (
        not isinstance(query, torch.Tensor)
        or not isinstance(key, torch.Tensor)
        or not isinstance(value, torch.Tensor)
    ):
        raise TypeError("Images must be a torch.Tensor".capitalize())

    else:
        weights = torch.matmul(
            input=query, other=torch.transpose(input=key, dim0=-2, dim1=-1)
        )
        weights = weights / math.sqrt(query.size(-1))

        attention = torch.softmax(input=weights, dim=-1)
        attention = torch.matmul(input=attention, other=value)

        return attention


if __name__ == "__main__":
    query = torch.randn((16, 8, 64, 64))
    key = torch.randn((16, 8, 64, 64))
    value = torch.randn((16, 8, 64, 64))

    attention = scale_dot_product(query=query, key=key, value=value)

    assert(attention.size()) == query.size(), "Attention must be the same size as query".capitalize()
    assert (attention.size()) == key.size(), "Attention must be the same size as key".capitalize()
    assert (attention.size()) == value.size(), "Attention must be the same size as value".capitalize()
