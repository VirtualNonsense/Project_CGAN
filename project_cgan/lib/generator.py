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


class DCGanGenerator(nn.Module):
    def __init__(self,
                 # number_of_gpus: int,
                 feature_map_size: int,
                 color_channels: int,
                 input_size: int,
                 kernel_size: Union[int, Tuple[int, int]] = 4,
                 stride: Union[int, Tuple[int, int]] = 2,
                 padding: Union[int, Tuple[int, int]] = 1,
                 bias: bool = False):
        super(DCGanGenerator, self).__init__()
        # self.number_of_gpus = number_of_gpus
        self.__input_size = input_size
        self.__kernel_size = (kernel_size, kernel_size) if type(kernel_size) is int else kernel_size
        self.__stride = (stride, stride) if type(stride) is int else stride
        self.__padding = (padding, padding) if type(padding) is int else padding
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # nn.ConvTranspose2d(input_size, feature_map_size * 16, kernel_size=self.__kernel_size, stride=self.__stride,
            #                    padding=(0, 0),
            #                    bias=bias),
            # nn.BatchNorm2d(feature_map_size * 16),
            # nn.ReLU(True),
            nn.ConvTranspose2d(input_size, feature_map_size * 8, kernel_size=self.__kernel_size, stride=self.__stride,
                               padding=(0, 0),
                               bias=bias),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(feature_map_size * 8, feature_map_size * 4, kernel_size=self.__kernel_size,
                               stride=self.__stride,
                               padding=self.__padding, bias=bias),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, kernel_size=self.__kernel_size,
                               stride=self.__stride,
                               padding=self.__padding, bias=bias),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(feature_map_size * 2, feature_map_size, kernel_size=self.__kernel_size,
                               stride=self.__stride,
                               padding=self.__padding, bias=bias),
            nn.BatchNorm2d(feature_map_size),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(feature_map_size, color_channels, kernel_size=self.__kernel_size, stride=self.__stride,
                               padding=self.__padding,
                               bias=bias),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    @property
    def input_size(self):
        return self.__input_size

    def forward(self, input_vector):
        return self.main(input_vector)


class CGanGenerator(nn.Module):
    """
    loosely based on:
    https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py
    """
    def __init__(self,
                 classes: int,
                 latent_dim: int,
                 img_shape: Optional[Tuple[int, int, int]] = None):
        super(CGanGenerator, self).__init__()
        if img_shape is None:
            img_shape = (3, 64, 64)

        self.img_shape = img_shape

        self.label_emb = nn.Embedding(classes, classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img

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