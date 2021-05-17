import os
from argparse import ArgumentParser
from collections import OrderedDict
from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

import pytorch_lightning as pl


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape: Tuple[int, int, int], output_dim: int):
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.width = img_shape[1]
        self.model = nn.Sequential(
            *block(latent_dim, self.width, normalize=False),
            *block(self.width, 2*self.width),
            *block(2*self.width, 4*self.width),
            *block(4*self.width, 8*self.width),
            nn.Linear(in_features=8*self.width, out_features=output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img
