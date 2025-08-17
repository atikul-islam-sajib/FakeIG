import os
import sys
import torch

sys.path.append("./src")

class Loader():
    def __init__(self, image_size: int = 224, split_size: float = 0.30, batch_size: int = 16):
        self.image_size = image_size
        self.split_size = split_size
        self.batch_size = batch_size
        
    def transform(self, type = "RGB"):
        pass
    
    def unzip_folder(self):
        pass
    
    def extract_features(self):
        pass
    
    def create_dataloader(self):
        pass
    
    
if __name__ == "__main__":
    loader = Loader(
        image_size=224,
        split_size=0.30,
        batch_size=16
    )