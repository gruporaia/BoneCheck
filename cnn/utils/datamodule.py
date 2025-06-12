import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from .dataset import CustomImageDataset

class DataModule(pl.LightningDataModule):
    def __init__(self, df_train, df_val, image_dir, batch_size=32):
        super().__init__()
        self.df_train = df_train
        self.df_val = df_val
        self.image_dir = image_dir
        self.batch_size = batch_size

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def setup(self, stage=None):
        self.train_dataset = CustomImageDataset(self.df_train, self.image_dir, transform=self.transform)
        self.val_dataset = CustomImageDataset(self.df_val, self.image_dir, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, persistent_workers=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, persistent_workers=True, num_workers=4, pin_memory=True)
