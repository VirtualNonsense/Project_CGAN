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


def __gen_block(input_size, output_size, kernel, stride, padding, negative_slope: float,
                inplace: bool, bias: bool, batch_norm: bool = True):
    if batch_norm:
        return [
            nn.Conv2d(input_size, output_size, kernel_size=kernel, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(output_size),
            nn.LeakyReLU(negative_slope, inplace=inplace)
        ]

    return [
        nn.Conv2d(input_size, output_size, kernel_size=kernel, stride=stride, padding=padding, bias=bias),
        nn.LeakyReLU(negative_slope, inplace=inplace)
    ]


def _gen_layers(layer_count: int, input_layer_size, output_layer_size, map_size, kernel, padding, stride,
                inplace: bool, bias: bool, negative_slope: float) -> List[any]:
    layers = __gen_block(input_layer_size, map_size, kernel=kernel, stride=stride, padding=padding,
                         inplace=inplace, negative_slope=negative_slope, bias=bias,
                         batch_norm=False)
    new_map_size = map_size * 2
    if layer_count > 2:
        for i in range(layer_count - 2):
            layers += __gen_block(map_size, new_map_size, kernel=kernel, stride=stride,
                                  padding=padding,
                                  inplace=inplace, negative_slope=negative_slope, bias=bias)
            map_size = new_map_size
            new_map_size *= 2
    layers += [
        nn.Conv2d(map_size, output_layer_size, kernel_size=kernel, stride=(1, 1), padding=(0, 0), bias=bias),
        nn.Sigmoid()
    ]
    return layers


class DCGanDiscriminator(nn.Module):
    def __init__(self,
                 # number_of_gpus: int,
                 feature_map_size: int,
                 input_channels: int,
                 kernel_size: Union[int, Tuple[int, int]] = 4,
                 stride: Union[int, Tuple[int, int]] = 2,
                 padding: Union[int, Tuple[int, int]] = 1,
                 bias: bool = False,
                 inplace: bool = True,
                 negative_slope: float = 0.2):
        super(DCGanDiscriminator, self).__init__()
        self.__kernel_size = (kernel_size, kernel_size) if type(kernel_size) is int else kernel_size
        self.__stride = (stride, stride) if type(stride) is int else stride
        self.__padding = (padding, padding) if type(padding) is int else padding
        # self.number_of_gpus = number_of_gpus
        self.main = nn.Sequential(
            *_gen_layers(5,
                         input_channels,
                         1,
                         feature_map_size,
                         self.__kernel_size,
                         self.__padding,
                         self.__stride,
                         bias=bias,
                         inplace=inplace,
                         negative_slope=negative_slope)
        )

    def forward(self, input_vector) -> torch.Tensor:
        return self.main(input_vector)


class CGanDiscriminator(nn.Module):
    """
    loosely based on:
    https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py
    """

    def __init__(self,
                 classes: int,
                 img_shape: Optional[Tuple[int, int, int]] = None):
        super(CGanDiscriminator, self).__init__()
        if img_shape is None:
            img_shape = (3, 64, 64)
        self.img_shape = img_shape
        self.label_embedding = nn.Embedding(classes, img_shape[1] * img_shape[2])

        self.model = nn.Sequential(
            nn.Linear(classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        embedding = self.label_embedding(labels).view(labels.shape[0], 1, self.img_shape[1], self.img_shape[2])
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        img = torch.cat([img, embedding], dim=1)  # N x C x img_size(H) x img_size(W)
        validity = self.model(d_in)
        return validity


class Discriminator(nn.Module):
    def __init__(self, img_shape: Tuple[int, int, int], output_dim: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(output_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


class CGanDis(nn.Module):
    """
    based on https://www.youtube.com/watch?v=Hp-jWm2SzR8
    """

    def __init__(self, channels_img, features_d, num_classes, img_size):
        super(CGanDis, self).__init__()
        self.img_size = img_size
        self.disc = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv2d(channels_img + 1, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0)
        )
        self.embed = nn.Embedding(num_classes, img_size * img_size)

    @staticmethod
    def _block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x, embedding], dim=1)  # N x C x img_size (H) x img_size (W)
        return self.disc(x)
