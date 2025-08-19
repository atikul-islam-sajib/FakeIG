import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append("./src")

try:
    from generator import Generator
    from discriminator import Discriminator
except ImportError:
    print("Import cannot be found".capitalize())

def load_dataloader():
    pass


def initialization(**kawrgs):
    lr = float(kawrgs["lr"])
    optimizer = str(kawrgs["optimizer"])
    beta1 = float(kawrgs["beta1"])
    beta2 = float(kawrgs["beta2"])
    momentum = float(kawrgs["momentum"])
    
    netG = Generator(
        latent_size=100,
        image_channels=3,
        image_size=224,
        patch_size=8,
        num_layers=4,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
        layer_norm_eps=1e-5,
    )

    netD = Discriminator(
        image_channels=3,
        image_size=224,
        patch_size=8,
        num_layers=4,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
        layer_norm_eps=1e-5,
    )
    
    if optimizer == "Adam":
        optimizerG = optim.Adam(params=netG.parameters(), lr=lr, betas=(beta1, beta2))
        optimizerD = optim.Adam(params=netD.parameters(), lr=lr, betas=(beta1, beta2))
    elif optimizer == "SGD":
        optimizerG = optim.SGD(params=netG.parameters(), lr=lr, momentum=0.9)
        optimizerD = optim.SGD(params=netD.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError("Optimizer not found".capitalize())
    
    criterion = nn.BCEWithLogitsLoss()
    
    return{
        "netG": netG,
        "netD": netD,
        "optimizerG": optimizerG,
        "optimizerD": optimizerD,
        "criterion": criterion
    }
    
    
if __name__ == "__main__":
    init = initialization(
        lr=0.0002,
        optimizer="Adam",
        beta1=0.5,
        beta2=0.999,
        momentum=0.9
    )
    
    assert init["netG"].__class__ == Generator
    assert init["netD"].__class__ == Discriminator
    assert init["optimizerG"].__class__ == optim.Adam
    assert init["optimizerD"].__class__ == optim.Adam
    assert init["criterion"].__class__ == nn.BCEWithLogitsLoss