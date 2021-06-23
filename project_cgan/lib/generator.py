from typing import *

import numpy as np
import torch
import torch.nn as nn


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
    def __init__(self, latent_dim, img_shape: Tuple[int, int, int], output_dim: int, num_classes: int, embed_size):
        super().__init__()
        self.img_shape = img_shape
        self.embed = nn.Embedding(num_classes, embed_size)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.width = img_shape[1]
        self.model = nn.Sequential(
            *block(latent_dim+embed_size, self.width, normalize=False),
            *block(self.width, 2 * self.width),
            *block(2 * self.width, 4 * self.width),
            *block(4 * self.width, 8 * self.width),
            nn.Linear(in_features=8 * self.width, out_features=output_dim),
            nn.Tanh()
        )

    def forward(self, z, labels):
        embedding = self.embed(labels)
        z = torch.cat([z, embedding], dim=1)
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class CGanGen(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g, num_classes, img_size, embed_size):
        super(CGanGen, self).__init__()
        self.img_size = img_size
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise+embed_size, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )
        self.embed = nn.Embedding(num_classes, embed_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, labels):
        # latent vector z: N x noise_dim x 1 x 1
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1)
        return self.net(x)
