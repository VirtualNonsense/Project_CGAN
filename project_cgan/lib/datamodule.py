"""
https://colab.research.google.com/github/PytorchLightning/pytorch-lightning/blob/master/notebooks/03-basic-gan.ipynb#scrollTo=DOY_nHu328g7
"""

import os
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import torchvision.datasets as datasets

import pytorch_lightning as pl

from dataloader import *
from typing import *


class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 256, num_workers: int = 12, set_image_size: int = 64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.Resize(set_image_size),
            transforms.CenterCrop(set_image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.size = set_image_size
        self.data_set = datasets.ImageFolder(root=self.data_dir, transform=self.transform)

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        # self.dims = (1, 28, 28)
        # self.num_classes = 10

    def train_dataloader(self):
        return DataLoader(dataset=self.data_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(dataset=self.data_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.data_set, batch_size=self.batch_size, num_workers=self.num_workers)
