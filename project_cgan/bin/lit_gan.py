# import os
# import torch
# from torch import nn
# import torch.nn.functional as F
# from torchvision import transforms
# from torchvision.datasets import MNIST
# from torch.utils.data import DataLoader, random_split
# import pytorch_lightning as pl
# from typing import *
#
#
# def __gen_discriminator_block(input_size, output_size, kernel, stride, padding, negative_slope: float,
#                               inplace: bool, bias: bool, batch_norm: bool = True):
#     if batch_norm:
#         return [
#             nn.Conv2d(input_size, output_size, kernel_size=kernel, stride=stride, padding=padding, bias=bias),
#             nn.BatchNorm2d(output_size),
#             nn.LeakyReLU(negative_slope, inplace=inplace)
#         ]
#
#     return [
#         nn.Conv2d(input_size, output_size, kernel_size=kernel, stride=stride, padding=padding, bias=bias),
#         nn.LeakyReLU(negative_slope, inplace=inplace)
#     ]
#
#
# def gen_discriminator_layers(layer_count: int, input_layer_size, output_layer_size, map_size, kernel, padding, stride,
#                              inplace: bool, bias: bool, negative_slope: float) -> List[
#     any]:
#     layers = __gen_discriminator_block(input_layer_size, map_size, kernel=kernel, stride=stride, padding=padding,
#                                        inplace=inplace, negative_slope=negative_slope, bias=bias, batch_norm=False)
#     new_map_size = map_size * 2
#     if layer_count > 2:
#         for i in range(layer_count - 2):
#             layers += __gen_discriminator_block(map_size, new_map_size, kernel=kernel, stride=stride,
#                                                 padding=padding,
#                                                 inplace=inplace, negative_slope=negative_slope, bias=bias)
#             map_size = new_map_size
#             new_map_size *= 2
#     layers += [
#         nn.Conv2d(map_size, output_layer_size, kernel_size=kernel, stride=(1, 1), padding=(0, 0), bias=bias),
#         nn.Sigmoid()
#     ]
#     return layers
#
#
# def __gen_generator_block(input_size, output_size, kernel, stride, padding, bias: bool, inplace: bool):
#     return [
#         nn.ConvTranspose2d(input_size, output_size, kernel_size=kernel,
#                            stride=stride,
#                            padding=padding,
#                            bias=bias),
#         nn.BatchNorm2d(output_size),
#         nn.ReLU(inplace=inplace),
#     ]
#
#
#
# def gen_generator_layers(layer_count: int, input_layer_size, output_layer_size, map_size, kernel, padding, stride,
#                          inplace: bool, bias: bool) -> List[any]:
#     layers = [__gen_generator_block()]
#
#     # nn.ConvTranspose2d(input_size, generator_feature_map_size * 8, kernel_size=self.__kernel_size, stride=self.__stride,
#     #                    padding=(0, 0),
#     #                    bias=bias),
#     # nn.BatchNorm2d(generator_feature_map_size * 8),
#     # nn.ReLU(True),
#     # # state size. (ngf*8) x 4 x 4
#     # nn.ConvTranspose2d(generator_feature_map_size * 8, generator_feature_map_size * 4, kernel_size=self.__kernel_size,
#     #                    stride=self.__stride,
#     #                    padding=self.__padding, bias=bias),
#     # nn.BatchNorm2d(generator_feature_map_size * 4),
#     # nn.ReLU(True),
#     # # state size. (ngf*4) x 8 x 8
#     # nn.ConvTranspose2d(generator_feature_map_size * 4, generator_feature_map_size * 2, kernel_size=self.__kernel_size,
#     #                    stride=self.__stride,
#     #                    padding=self.__padding, bias=bias),
#     # nn.BatchNorm2d(generator_feature_map_size * 2),
#     # nn.ReLU(True),
#     # # state size. (ngf*2) x 16 x 16
#     # nn.ConvTranspose2d(generator_feature_map_size * 2, generator_feature_map_size, kernel_size=self.__kernel_size,
#     #                    stride=self.__stride,
#     #                    padding=self.__padding, bias=bias),
#     # nn.BatchNorm2d(generator_feature_map_size),
#     # nn.ReLU(True),
#     # # state size. (ngf) x 32 x 32
#     # nn.ConvTranspose2d(generator_feature_map_size, color_channels, kernel_size=self.__kernel_size, stride=self.__stride,
#     #                    padding=self.__padding,
#     #                    bias=bias),
#     # nn.Tanh()
#     return layers
#
#
# class LitGan(pl.LightningModule):
#     def __init__(self,
#                  discriminator_feature_map_size: int,
#                  generator_feature_map_size: int,
#                  color_channels: int,
#                  input_size: int,
#                  kernel_size: Tuple[int, int] = (4, 4),
#                  stride: Tuple[int, int] = (2, 2),
#                  padding: Tuple[int, int] = (1, 1),
#                  bias: bool = False,
#                  inplace: bool = True,
#                  negative_slope: float = 0.2):
#         super().__init__()
#         self.discriminator = nn.Sequential(
#             *gen_discriminator_layers(5,
#                                       color_channels,
#                                       1,
#                                       discriminator_feature_map_size,
#                                       kernel_size,
#                                       padding,
#                                       stride,
#                                       bias=bias,
#                                       inplace=inplace,
#                                       negative_slope=negative_slope)
#         )
#
#         self.generator = nn.Sequential(
#             nn.ConvTranspose2d(input_size, generator_feature_map_size * 8, kernel_size=self.__kernel_size,
#                                stride=self.__stride,
#                                padding=(0, 0),
#                                bias=bias),
#             nn.BatchNorm2d(generator_feature_map_size * 8),
#             nn.ReLU(True),
#             # state size. (ngf*8) x 4 x 4
#             nn.ConvTranspose2d(generator_feature_map_size * 8, generator_feature_map_size * 4,
#                                kernel_size=self.__kernel_size,
#                                stride=self.__stride,
#                                padding=self.__padding, bias=bias),
#             nn.BatchNorm2d(generator_feature_map_size * 4),
#             nn.ReLU(True),
#             # state size. (ngf*4) x 8 x 8
#             nn.ConvTranspose2d(generator_feature_map_size * 4, generator_feature_map_size * 2,
#                                kernel_size=self.__kernel_size,
#                                stride=self.__stride,
#                                padding=self.__padding, bias=bias),
#             nn.BatchNorm2d(generator_feature_map_size * 2),
#             nn.ReLU(True),
#             # state size. (ngf*2) x 16 x 16
#             nn.ConvTranspose2d(generator_feature_map_size * 2, generator_feature_map_size,
#                                kernel_size=self.__kernel_size,
#                                stride=self.__stride,
#                                padding=self.__padding, bias=bias),
#             nn.BatchNorm2d(generator_feature_map_size),
#             nn.ReLU(True),
#             # state size. (ngf) x 32 x 32
#             nn.ConvTranspose2d(generator_feature_map_size, color_channels, kernel_size=self.__kernel_size,
#                                stride=self.__stride,
#                                padding=self.__padding,
#                                bias=bias),
#             nn.Tanh()
#
#         )
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch.nn import functional as F

from project_cgan.lib.discriminator import GanDiscriminator
from project_cgan.lib.generator import GanGenerator
from typing import *


class GAN(pl.LightningModule):
    """
    Vanilla GAN implementation.
    Example::
        from pl_bolts.models.gans import GAN
        m = GAN()
        Trainer(gpus=2).fit(m)
    Example CLI::
        # mnist
        python  basic_gan_module.py --gpus 1
        # imagenet
        python  basic_gan_module.py --gpus 1 --dataset 'imagenet2012'
        --data_dir /path/to/imagenet/folder/ --meta_dir ~/path/to/meta/bin/folder
        --batch_size 256 --learning_rate 0.0001
    """

    def __init__(
            self,
            color_channels: int,
            discriminator_feature_map_size: int,
            generator_feature_map_size: int,
            generator_input_channels: int,
            learning_rate: float = 0.0002,
            **kwargs
    ):
        """
        Args:
            color_channels: number of channels of an image
            input_height: image height
            input_width: image width
            latent_dim: emb dim for encoder
            learning_rate: the learning rate
        """
        super().__init__()
        self.learning_rate = learning_rate
        # makes self.hparams under the hood and saves to ckpt
        # self.img_dim = (input_channels, input_height, input_width)

        # networks
        self.discriminator = GanDiscriminator(
            feature_map_size=discriminator_feature_map_size,
            input_channels=color_channels
        )

        self.generator = GanGenerator(
            feature_map_size=generator_feature_map_size,
            color_channels=color_channels,
            input_size=generator_input_channels
        )
        self.save_hyperparameters()

    def forward(self, z):
        """
        Generates an image given input noise z
        Example::
            z = torch.rand(batch_size, latent_dim)
            gan = GAN.load_from_checkpoint(PATH)
            img = gan(z)
        """
        return self.generator(z)

    def generator_loss(self, x):
        # sample noise
        z = torch.randn(x.shape[0], self.hparams.latent_dim, device=self.device)
        y = torch.ones(x.size(0), 1, device=self.device)

        # generate images
        generated_imgs = self(z)

        D_output = self.discriminator(generated_imgs)

        # ground truth result (ie: all real)
        g_loss = F.binary_cross_entropy(D_output, y)

        return g_loss

    def discriminator_loss(self, x):
        # train discriminator on real
        b = x.size(0)
        x_real = x.view(b, -1)
        y_real = torch.ones(b, 1, device=self.device)

        # calculate real score
        D_output = self.discriminator(x_real)
        D_real_loss = F.binary_cross_entropy(D_output, y_real)

        # train discriminator on fake
        z = torch.randn(b, self.hparams.latent_dim, device=self.device)
        x_fake = self(z)
        y_fake = torch.zeros(b, 1, device=self.device)

        # calculate fake score
        D_output = self.discriminator(x_fake)
        D_fake_loss = F.binary_cross_entropy(D_output, y_fake)

        # gradient backprop & optimize ONLY D's parameters
        D_loss = D_real_loss + D_fake_loss

        return D_loss

    def training_step(self,
                      batch: Union[Tuple[torch.tensor, torch.tensor], List[torch.tensor]],
                      batch_idx: int,
                      optimizer_idx: int):
        x, _ = batch

        # train generator
        result = None
        if optimizer_idx == 0:
            result = self.generator_step(x)

        # train discriminator
        if optimizer_idx == 1:
            result = self.discriminator_step(x)

        return result

    def generator_step(self, x):
        g_loss = self.generator_loss(x)

        # log to prog bar on each step AND for the full epoch
        # use the generator loss for checkpointing
        self.log('g_loss', g_loss, on_epoch=True, prog_bar=True)
        return g_loss

    def discriminator_step(self, x):
        # Measure discriminator's ability to classify real from generated samples
        d_loss = self.discriminator_loss(x)

        # log to prog bar on each step AND for the full epoch
        self.log('d_loss', d_loss, on_epoch=True, prog_bar=True)
        return d_loss

    def configure_optimizers(self):
        lr = self.hparams.learning_rate

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return [opt_g, opt_d], []

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0002, help="adam: learning rate")
        parser.add_argument(
            '--adam_b1', type=float, default=0.5, help="adam: decay of first order momentum of gradient"
        )
        parser.add_argument(
            '--adam_b2', type=float, default=0.999, help="adam: decay of first order momentum of gradient"
        )
        parser.add_argument('--latent_dim', type=int, default=100, help="generator embedding dim")
        return parser


def cli_main(args=None):

    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument("--dataset", default="mnist", type=str, help="mnist, cifar10, stl10, imagenet")
    script_args, _ = parser.parse_known_args(args)

    if script_args.dataset == "mnist":
        dm_cls = MNISTDataModule
    elif script_args.dataset == "cifar10":
        dm_cls = CIFAR10DataModule
    elif script_args.dataset == "stl10":
        dm_cls = STL10DataModule
    elif script_args.dataset == "imagenet":
        dm_cls = ImagenetDataModule

    parser = dm_cls.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = GAN.add_model_specific_args(parser)
    args = parser.parse_args(args)

    dm = dm_cls.from_argparse_args(args)
    model = GAN(*dm.size(), **vars(args))
    callbacks = [TensorboardGenerativeModelImageSampler(), LatentDimInterpolator(interpolate_epoch_interval=5)]
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, progress_bar_refresh_rate=20)
    trainer.fit(model, datamodule=dm)
    return dm, model, trainer

if __name__ == '__main__':
    dm, model, trainer = cli_main()

