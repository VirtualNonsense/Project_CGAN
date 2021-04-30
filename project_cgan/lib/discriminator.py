import torch
import torch.nn as nn
from typing import *
import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn.functional as F
import torch


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


class CGanGenerator(nn.Module):
    """
    loosely based on:
    https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py
    """
    def __init__(self,
                 classes: int,
                 latent_dim: int = 100,
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
