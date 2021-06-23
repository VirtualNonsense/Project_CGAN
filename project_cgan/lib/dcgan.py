from typing import *
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter


class Generator(nn.Module):
    def __init__(self,
                 input_dim: int,
                 filter_sizes: List[int],
                 output_dim: int,
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
        self.hidden_layer = torch.nn.Sequential()
        self.output_layer = torch.nn.Sequential()

        # generator representation
        # each class has it's own 1 layer within this tensor.
        for i in range(len(filter_sizes)):
            # Deconvolutional layer
            if i == 0:
                # For input
                input_deconv = torch.nn.ConvTranspose2d(input_dim,
                                                        int(filter_sizes[i]),
                                                        kernel_size=self.__kernel_size,
                                                        stride=(1, 1),
                                                        padding=(0, 0))
                self.image_layer.add_module('input_deconv', input_deconv)

                # Initializer
                torch.nn.init.normal_(input_deconv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(input_deconv.bias, 0.0)

                # Batch normalization
                self.image_layer.add_module('input_bn', torch.nn.BatchNorm2d(int(filter_sizes[i])))

                # Activation
                self.image_layer.add_module('input_act', torch.nn.ReLU())

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

        # Deconvolutional layer
        out = torch.nn.ConvTranspose2d(filter_sizes[-1], output_dim,
                                       kernel_size=self.__kernel_size,
                                       stride=self.__stride,
                                       padding=self.__padding)
        self.output_layer.add_module('out', out)
        # Initializer
        torch.nn.init.normal_(out.weight, mean=0.0, std=0.02)
        torch.nn.init.constant_(out.bias, 0.0)
        # Activation
        self.output_layer.add_module('act', torch.nn.Tanh())

    def forward(self, noise):
        h1 = self.image_layer(noise)
        h = self.hidden_layer(h1)
        out = self.output_layer(h)
        return out


class Discriminator(nn.Module):
    def __init__(self,
                 input_dim,
                 filter_sizes: List[int],
                 output_dim: int,
                 kernel_size: Union[int, Tuple[int, int]] = 4,
                 stride: Union[int, Tuple[int, int]] = 2,
                 padding: Union[int, Tuple[int, int]] = 1):
        super().__init__()
        self.__kernel_size = (kernel_size, kernel_size) if type(kernel_size) is int else kernel_size
        self.__stride = (stride, stride) if type(stride) is int else stride
        self.__padding = (padding, padding) if type(padding) is int else padding

        self.image_layer = torch.nn.Sequential()
        self.hidden_layer = torch.nn.Sequential()
        self.output_layer = torch.nn.Sequential()

        for i in range(len(filter_sizes)):
            # Convolutional layer
            if i == 0:
                # For input
                input_conv = torch.nn.Conv2d(input_dim, int(filter_sizes[i]),
                                             kernel_size=self.__kernel_size,
                                             stride=self.__stride,
                                             padding=self.__padding)
                self.image_layer.add_module('input_conv', input_conv)

                # Initializer
                torch.nn.init.normal_(input_conv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(input_conv.bias, 0.0)

                # Activation
                self.image_layer.add_module('input_act', torch.nn.LeakyReLU(0.2))

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
        # Convolutional layer
        out = torch.nn.Conv2d(filter_sizes[-1],
                              output_dim,
                              kernel_size=self.__kernel_size, stride=(1, 1), padding=(0, 0))
        self.output_layer.add_module('out', out)
        # Initializer
        torch.nn.init.normal_(out.weight, mean=0.0, std=0.02)
        torch.nn.init.constant_(out.bias, 0.0)
        # Activation
        self.output_layer.add_module('act', torch.nn.Sigmoid())

    def forward(self, images):
        h1 = self.image_layer(images)
        h = self.hidden_layer(h1)
        out = self.output_layer(h)
        return out


class DCGAN(pl.LightningModule):

    def __init__(self,
                 input_dim: int,
                 filter_sizes: List[int],
                 color_channels: int,
                 image_size: int,
                 device: torch.device,
                 writer: Optional[SummaryWriter] = None,
                 image_intervall=10,
                 tensorboard_image_rows=10,
                 tensorboard_image_columns=8,
                 batch_size: int = 128):
        super().__init__()
        self.writer = writer
        self.tensorboard_image_columns = tensorboard_image_columns
        self.tensorboard_images_rows = tensorboard_image_rows
        self.image_intervall = image_intervall
        self.used_device = device
        self.image_size = image_size
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.loc_scale = (0, 1)
        self.filter_sizes = filter_sizes

        self.generator = Generator(

            input_dim=input_dim,
            filter_sizes=filter_sizes,
            output_dim=color_channels,
        )
        self.discriminator = Discriminator(
            input_dim=color_channels,
            filter_sizes=filter_sizes[::-1],
            output_dim=1,
        )
        self.criterion = nn.BCELoss()
        self.validation_z = torch.rand(batch_size, input_dim)
        self.sample_noise: Union[torch.Tensor, None] = None

    def forward(self, z):
        """
        Generates an image using the generator
        given input noise z
        """
        return self.generator(z)

    def generator_step(self):
        """
        Training step for generator
        1. Sample random noise
        2. Pass noise to generator to
           generate images
        3. Classify generated images using
           the discriminator
        4. Backprop loss
        """

        if self.sample_noise is None:
            # saving noise and lables for
            fixed_noise = torch.tensor(
                np.random.normal(self.loc_scale[0], self.loc_scale[1],
                                 (self.tensorboard_images_rows * self.tensorboard_image_columns, self.generator.latent_dim, 1, 1)),
                device=self.used_device, dtype=torch.float)
            self.sample_noise = fixed_noise

        z = torch.tensor(
            np.random.normal(self.loc_scale[0], self.loc_scale[1], (self.batch_size, self.generator.latent_dim, 1, 1)),
            device=self.used_device, dtype=torch.float)

        # Generate images
        generated_imgs = self(z)
        # Classify generated image using the discriminator
        d_g_z: torch.tensor = self.discriminator(generated_imgs)

        d_output = d_g_z.reshape(-1)

        d_ref = torch.ones(self.batch_size, device=self.used_device)
        g_loss = self.criterion(d_output,
                                d_ref)
        if self.writer is not None:
            self.writer.add_scalar("Generator Loss", g_loss, self.global_step)
            self.writer.add_scalar("d(g(z))", d_g_z.view(-1).mean().item(), self.global_step)
        return g_loss

    def discriminator_step(self, x):
        """
        Training step for discriminator
        1. Get actual images and labels
        2. Predict probabilities of actual images and get BCE loss
        3. Get fake images from generator
        4. Predict probabilities of fake images and get BCE loss
        5. Combine loss from both and backprop
        """

        # Real images
        d_ref_r = torch.ones((x.shape[0]), device=self.used_device)
        d_i = self.discriminator(x).reshape(-1)
        loss_real = self.criterion(d_i,
                                   d_ref_r)

        # Fake images
        z = torch.tensor(
            np.random.normal(self.loc_scale[0], self.loc_scale[1], (x.shape[0], self.generator.latent_dim, 1, 1)),
            device=self.used_device, dtype=torch.float)

        generated_imgs = self(z)
        d_g_z = self.discriminator(generated_imgs)
        d_output = d_g_z.reshape(-1)
        d_zeros = torch.zeros((x.shape[0]), device=self.used_device)
        loss_fake = self.criterion(d_output,
                                   d_zeros)
        if self.writer is not None:
            self.writer.add_scalar("Discriminator Loss", loss_fake + loss_real, self.global_step)
            self.writer.add_scalar("d(i)", d_i.view(-1).mean().item(), self.global_step)
        return loss_real + loss_fake

    def training_step(self, batch, batch_idx, optimizer_idx):
        X, y = batch
        loss = None
        # train generator
        if optimizer_idx == 0:
            loss = self.generator_step()

        # train discriminator
        if optimizer_idx == 1:
            loss = self.discriminator_step(X)

        return loss

    def configure_optimizers(self):
        g_optimizer = torch.optim.Adam(self.generator.parameters(), betas=(0.5, 0.999), lr=0.0002)
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), betas=(0.5, 0.999), lr=0.0002)
        return g_optimizer, d_optimizer

    def on_epoch_end(self) -> None:
        imgs = self(self.sample_noise)
        d_g_z = self.discriminator(imgs).reshape(-1)
        g_loss = self.criterion(d_g_z, torch.ones(d_g_z.shape[0], device=self.device))
        self.log("g_loss", g_loss)
        if self.writer is not None:
            if self.current_epoch % self.image_intervall == 0:
                # denormalize
                imgs = (imgs + 1) / 2
                grid = torchvision.utils.make_grid(imgs, nrow=self.tensorboard_image_columns)
                self.writer.add_image('images', grid, global_step=self.current_epoch)
            self.writer.close()

