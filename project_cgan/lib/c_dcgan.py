"""
adapted approach from https://github.com/togheppi/cDCGAN
"""

import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms, datasets
import pytorch_lightning as pl
import transform
import numpy as np

from pytorch_lightning.callbacks import ModelCheckpoint

import datamodule
from typing import *


class Generator(nn.Module):
    def __init__(self,
                 input_dim,
                 label_dim,
                 filter_sizes: List[int],
                 output_dim,
                 kernel_size: Union[int, Tuple[int, int]] = 4,
                 stride: Union[int, Tuple[int, int]] = 2,
                 padding: Union[int, Tuple[int, int]] = 1):
        super().__init__()
        self.latent_dim = input_dim
        self.__kernel_size = (kernel_size, kernel_size) if type(kernel_size) is int else kernel_size
        self.__stride = (stride, stride) if type(stride) is int else stride
        self.__padding = (padding, padding) if type(padding) is int else padding
        # Hidden layers
        self.image_layer = torch.nn.Sequential()
        self.label_layer = torch.nn.Sequential()
        self.hidden_layer = torch.nn.Sequential()
        for i in range(len(filter_sizes)):
            # Deconvolutional layer
            if i == 0:
                # For input
                input_deconv = torch.nn.ConvTranspose2d(input_dim,
                                                        int(filter_sizes[i] / 2),
                                                        kernel_size=self.__kernel_size,
                                                        stride=(1,1),
                                                        padding=(0, 0))
                self.image_layer.add_module('input_deconv', input_deconv)

                # Initializer
                torch.nn.init.normal_(input_deconv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(input_deconv.bias, 0.0)

                # Batch normalization
                self.image_layer.add_module('input_bn', torch.nn.BatchNorm2d(int(filter_sizes[i] / 2)))

                # Activation
                self.image_layer.add_module('input_act', torch.nn.ReLU())

                # For label
                label_deconv = torch.nn.ConvTranspose2d(label_dim, int(filter_sizes[i] / 2),
                                                        kernel_size=self.__kernel_size,
                                                        stride=self.__stride)
                self.label_layer.add_module('label_deconv', label_deconv)

                # Initializer
                torch.nn.init.normal_(label_deconv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(label_deconv.bias, 0.0)

                # Batch normalization
                self.label_layer.add_module('label_bn', torch.nn.BatchNorm2d(int(filter_sizes[i] / 2)))

                # Activation
                self.label_layer.add_module('label_act', torch.nn.ReLU())
            else:
                deconv = torch.nn.ConvTranspose2d(filter_sizes[i - 1], filter_sizes[i],
                                                  kernel_size=self.__kernel_size,
                                                  stride=self.__stride,
                                                  padding=self.__padding)

                deconv_name = 'deconv' + str(i + 1)
                self.hidden_layer.add_module(deconv_name, deconv)

                # Initializer
                torch.nn.init.normal_(deconv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(deconv.bias, 0.0)

                # Batch normalization
                bn_name = 'bn' + str(i + 1)
                self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(filter_sizes[i]))

                # Activation
                act_name = 'act' + str(i + 1)
                self.hidden_layer.add_module(act_name, torch.nn.ReLU())

            # Output layer
            self.output_layer = torch.nn.Sequential()
            # Deconvolutional layer
            out = torch.nn.ConvTranspose2d(filter_sizes[i], output_dim,
                                           kernel_size=self.__kernel_size,
                                           stride=self.__stride,
                                           padding=self.__padding)
            self.output_layer.add_module('out', out)
            # Initializer
            torch.nn.init.normal_(out.weight, mean=0.0, std=0.02)
            torch.nn.init.constant_(out.bias, 0.0)
            # Activation
            self.output_layer.add_module('act', torch.nn.Tanh())

    def forward(self, noise, labels):
        h1 = self.image_layer(noise)
        h2 = self.label_layer(labels)
        x = torch.cat([h1, h2], 1)
        h = self.hidden_layer(x)
        out = self.output_layer(h)
        return out


class Discriminator(nn.Module):
    def __init__(self,
                 input_dim,
                 label_dim,
                 filter_sizes: List[int],
                 output_dim,
                 kernel_size: Union[int, Tuple[int, int]] = 4,
                 stride: Union[int, Tuple[int, int]] = 2,
                 padding: Union[int, Tuple[int, int]] = 1):
        super().__init__()
        self.__kernel_size = (kernel_size, kernel_size) if type(kernel_size) is int else kernel_size
        self.__stride = (stride, stride) if type(stride) is int else stride
        self.__padding = (padding, padding) if type(padding) is int else padding

        self.image_layer = torch.nn.Sequential()
        self.label_layer = torch.nn.Sequential()
        self.hidden_layer = torch.nn.Sequential()
        for i in range(len(filter_sizes)):
            # Convolutional layer
            if i == 0:
                # For input
                input_conv = torch.nn.Conv2d(input_dim, int(filter_sizes[i] / 2),
                                             kernel_size=self.__kernel_size,
                                             stride=self.__stride,
                                             padding=self.__padding)
                self.image_layer.add_module('input_conv', input_conv)

                # Initializer
                torch.nn.init.normal_(input_conv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(input_conv.bias, 0.0)

                # Activation
                self.image_layer.add_module('input_act', torch.nn.LeakyReLU(0.2))

                # For label
                label_conv = torch.nn.Conv2d(label_dim, int(filter_sizes[i] / 2),
                                             kernel_size=self.__kernel_size,
                                             stride=self.__stride,
                                             padding=self.__padding)
                self.label_layer.add_module('label_conv', label_conv)

                # Initializer
                torch.nn.init.normal_(label_conv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(label_conv.bias, 0.0)

                # Activation
                self.label_layer.add_module('label_act', torch.nn.LeakyReLU(0.2))
            else:
                conv = torch.nn.Conv2d(filter_sizes[i - 1], filter_sizes[i],
                                       kernel_size=self.__kernel_size,
                                       stride=self.__stride,
                                       padding=self.__padding)

                conv_name = 'conv' + str(i + 1)
                self.hidden_layer.add_module(conv_name, conv)

                # Initializer
                torch.nn.init.normal_(conv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(conv.bias, 0.0)

                # Batch normalization
                bn_name = 'bn' + str(i + 1)
                self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(filter_sizes[i]))

                # Activation
                act_name = 'act' + str(i + 1)
                self.hidden_layer.add_module(act_name, torch.nn.LeakyReLU(0.2))

            # Output layer
            self.output_layer = torch.nn.Sequential()
            # Convolutional layer
            out = torch.nn.Conv2d(filter_sizes[i],
                                  output_dim,
                                  kernel_size=self.__kernel_size, stride=self.__stride, padding=(0, 0))
            self.output_layer.add_module('out', out)
            # Initializer
            torch.nn.init.normal_(out.weight, mean=0.0, std=0.02)
            torch.nn.init.constant_(out.bias, 0.0)
            # Activation
            self.output_layer.add_module('act', torch.nn.Sigmoid())

    def forward(self, images, labels):
        h1 = self.image_layer(images)
        h2 = self.label_layer(labels)
        images = torch.cat([h1, h2], 1)
        h = self.hidden_layer(images)
        out = self.output_layer(h)
        return out


class CDCGAN(pl.LightningModule):

    def __init__(self,
                 input_dim: int,
                 amount_classes: int,
                 filter_sizes: List[int],
                 color_channels: int,
                 device: torch.device,
                 batch_size: int = 128):
        super().__init__()
        self.save_hyperparameters()
        self.used_device = device
        self.amount_classes = amount_classes
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.generator = Generator(
            input_dim=input_dim,
            label_dim=amount_classes,
            filter_sizes=filter_sizes,
            output_dim=color_channels
        )
        self.discriminator = Discriminator(
            input_dim=color_channels,
            label_dim=amount_classes,
            filter_sizes=filter_sizes,
            output_dim=1,
        )
        self.validation_z = torch.rand(batch_size, input_dim)
        self.sample_noise = None

    def forward(self, z, labels):
        """
        Generates an image using the generator
        given input noise z and labels y
        """
        return self.generator(z, labels)

    def generator_step(self, x):
        """
        Training step for generator
        1. Sample random noise and labels
        2. Pass noise and labels to generator to
           generate images
        3. Classify generated images using
           the discriminator
        4. Backprop loss
        """

        # Sample random noise and labels

        z = torch.tensor(np.random.normal(0, 1, (batch_size, self.generator.latent_dim, 1, 1)), device=device, dtype=torch.float)
        # z = torch.randn(x.shape[0], self.input_dim, device=self.used_device)
        # y = torch.randint(0, self.amount_classes, size=(x.shape[0],), device=self.used_device)
        y = torch.tensor(np.random.randint(0, self.amount_classes, size=(self.batch_size, amount_classes, 1, 1)), device=device, dtype=torch.float)
        if self.sample_noise is None:
            # saving noise and lables for
            self.sample_noise = (z, y)

        # Generate images
        generated_imgs = self(z, y)
        z = torch.reshape(generated_imgs, (-1, 3, 64, 64))[:64]
        # log sampled images
        # Log generated images

        # Classify generated image using the discriminator
        d_output = torch.squeeze(self.discriminator(generated_imgs, y))

        # Backprop loss. We want to maximize the discriminator's
        # loss, which is equivalent to minimizing the loss with the true
        # labels flipped (i.e. y_true=1 for fake images). We do this
        # as PyTorch can only minimize a function instead of maximizing
        g_loss = nn.BCELoss()(d_output,
                              torch.ones(x.shape[0], device=self.used_device))
        if self.current_epoch % 7 == 0:
            imgs = self(self.sample_noise[0], self.sample_noise[1])
            imgs = torch.reshape(generated_imgs, (-1, 3, 64, 64))[:64]
            grid = torchvision.utils.make_grid(imgs)
            writer.add_image('images', grid, global_step=self.current_epoch)
            writer.add_scalar("Generator Loss", g_loss, self.current_epoch)
        return g_loss

    def discriminator_step(self, x, y):
        """
        Training step for discriminator
        1. Get actual images and labels
        2. Predict probabilities of actual images and get BCE loss
        3. Get fake images from generator
        4. Predict probabilities of fake images and get BCE loss
        5. Combine loss from both and backprop
        """

        # Real images
        d_output = torch.squeeze(self.discriminator(x, y))
        loss_real = nn.BCELoss()(d_output,
                                 torch.ones(x.shape[0], device=self.used_device))

        # Fake images
        z = torch.randn(x.shape[0], 100, device=self.used_device)
        y = torch.randint(0, self.amount_classes, size=(x.shape[0],), device=self.used_device)

        generated_imgs = self(z, y)
        d_output = torch.squeeze(self.discriminator(generated_imgs, y))
        loss_fake = nn.BCELoss()(d_output,
                                 torch.zeros(x.shape[0], device=self.used_device))

        writer.add_scalar("Discriminator Loss", loss_fake + loss_real, self.current_epoch)
        return loss_real + loss_fake

    def training_step(self, batch, batch_idx, optimizer_idx):
        X, y = batch
        loss = None
        # train generator
        if optimizer_idx == 0:
            loss = self.generator_step(X)

        # train discriminator
        if optimizer_idx == 1:
            loss = self.discriminator_step(X, y)

        return loss

    def configure_optimizers(self):
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002)
        return [g_optimizer, d_optimizer], []

    def on_epoch_end(self) -> None:
        # z = self.validation_z.type_as(self.generator.model[0][0].weight)[:64]
        # # log sampled images
        # sample_imgs = self(z)
        # # Log generated images
        # grid = torchvision.utils.make_grid(sample_imgs)
        # writer.add_image('images', grid, global_step=self.current_epoch)
        writer.close()


if __name__ == "__main__":
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./checkpoints',
        save_top_k=3,
        mode='min'
    )
    color_channels = 3
    amount_classes = 12
    batch_size = 512

    image_size = 64
    label_dim = 2
    latent_dim = 100
    num_filters = [1024, 512, 256, 128]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path = os.environ['CGAN_SORTED']
    print(f"grabbing trainingsdata from: {path}")

    # mnist_transforms = transforms.Compose([transforms.ToTensor(),
    #                                        transforms.Normalize(mean=[0.5], std=[0.5]),
    #                                        transforms.Lambda(lambda x: x.view(-1, 784)),
    #                                        transforms.Lambda(lambda x: torch.squeeze(x))
    #                                        ])
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    data = datasets.ImageFolder(root=path, transform=transform)
    # data = datasets.MNIST(root='../data/MNIST', download=True, transform=mnist_transforms)

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=6)
    dm = datamodule.DataModule(path)
    writer = SummaryWriter()
    model = CDCGAN(
        input_dim=latent_dim,
        amount_classes=amount_classes,
        filter_sizes=num_filters,
        color_channels=color_channels,
        device=device,
        batch_size=batch_size
    )

    trainer = pl.Trainer(
        max_epochs=5000,
        gpus=1 if torch.cuda.is_available() else 0,
        progress_bar_refresh_rate=5,
        profiler='simple',
        callbacks=[checkpoint_callback],
    )
    # trainer.tune(model, dm)
    trainer.fit(model, dataloader)
