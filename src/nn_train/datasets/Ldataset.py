import os

import lightning as L
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
from dataset import CustomDataset
from pathlib import Path

transform = transforms.Compose([
                transforms.Resize((640, 640)),
                transforms.ToTensor(),
            ])

class MyDataModule(L.LightningDataModule):
    def __init__(self, data_dir: Path) -> None:
        super().__init__()
        self.path_to_dataset = data_dir
        self.num_workers = os.cpu_count() - 1

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = None):

        self.dataset_full = CustomDataset(self.path_to_dataset, transform=transform)

        if stage == "fit":
            self.dataset_train, self.dataset_val = random_split(
                self.dataset_full,
                [0.8, 0.2],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=16,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=16,
            num_workers=self.num_workers,
            shuffle=False,
        )
