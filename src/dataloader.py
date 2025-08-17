import os
import sys
import cv2
import joblib
import zipfile
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

sys.path.append("./src")


class Loader:
    def __init__(
        self,
        image_size: int = 224,
        split_size: float = 0.30,
        batch_size: int = 16,
        type="RGB",
    ):
        self.image_size = image_size
        self.split_size = split_size
        self.batch_size = batch_size
        self.type = type

        self.images = list()

    def transform(self, type="RGB"):
        if type == "RGB":
            return transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.CenterCrop((self.image_size, self.image_size)),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        elif type == "GRAY":
            return transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.CenterCrop((self.image_size, self.image_size)),
                    transforms.Normalize(mean=[0.485], std=[0.229]),
                ]
            )

    def unzip_folder(self):
        if not os.path.exists("./data/processed"):
            os.makedirs("./data/processed")

        with zipfile.ZipFile(file="./data/raw/images.zip", mode="r") as images:
            images.extractall(path="./data/processed")

    def extract_features(self):
        processed_images = os.path.join("./data/processed", "images")

        for _, image in enumerate(tqdm(os.listdir(processed_images))):
            image = os.path.join(processed_images, image)

            if not image.endswith((".jpg", ".png", ".jpeg")):
                continue

            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = Image.fromarray(image)
            image = self.transform(type=self.type)(image)

            self.images.append(image)

        train, test = (
            self.images[0 : int(len(self.images) * self.split_size)],
            self.images[int(len(self.images) * self.split_size) :],
        )

        return train, test

    def create_dataloader(self):
        train, test = self.extract_features()

        train_dataloader = DataLoader(
            dataset=list(zip(train, [1] * len(train))),
            batch_size=self.batch_size,
            shuffle=True,
        )
        valid_dataloader = DataLoader(
            dataset=list(zip(test, [0] * len(test))),
            batch_size=self.batch_size,
            shuffle=True,
        )

        if not os.path.exists("./data/processed"):
            os.makedirs("./data/processed")

        for filename, data in [
            ("train_dataloader.pkl", train_dataloader),
            ("valid_dataloader.pkl", valid_dataloader),
        ]:
            joblib.dump(data, os.path.join("./data/processed", filename))


if __name__ == "__main__":
    loader = Loader(image_size=224, split_size=0.80, batch_size=16)
    loader.unzip_folder()
    loader.extract_features()
    loader.create_dataloader()
