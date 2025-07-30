import cv2
from matplotlib.pyplot import plot as plt
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CustomDataset(Dataset):
    """Custom dataset for loading images and corresponding labels from separate folders.

    Directory structure:
        main_folder/
            images/
                img1.png
                img2.png
                ...
            labels/
                img1.txt
                img2.txt`
                ...

    Example:
        >>> path = "path/to/data"
        >>> dataset = CustomDataset(path)
        >>> sample = dataset[0]
        >>> print(len(dataset))
        >>> print(dataset[1])
    """

    def __init__(self, data_path: Path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.image_paths = []
        self.label_paths = []
        self.setup_data()

        assert self.image_paths, "Images not found"
        assert self.label_paths, "Labels not found"

    def setup_data(self):
        data_dirs = Path(self.data_path)
        self.image_paths = sorted((data_dirs / "images").glob("*.png"))
        self.label_paths = sorted((data_dirs / "labels").glob("*.txt"))

    def __len__(self):
        if len(self.image_paths) != len(self.label_paths):
            raise ValueError("image paths does not match with labels for length")
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx])

        with open(self.label_paths[idx], "r") as f:
            coords = f.read()

        coords = list(map(float, coords.split()))
        label = torch.tensor(coords, dtype=torch.float32)
        label = label.view(-1, 2)

        if self.transform:
            img = self.transform(img)

        else:
            img = transforms.ToTensor()(img)

        return img, label

