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
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import generator
import discriminator
import datamodule


class GAN(pl.LightningModule):
    def __init__(self,
                 channels, width, heigth, latent_dim: int = 100, lr: float = 0.002, b1: float = 0.5, b2: float = 0.999,
                 batch_size: int = 64, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.save_hyperparameters()

        # networks
        data_shape = (channels, width, heigth)
        self.generator = generator.Generator(
            latent_dim=self.hparams.latent_dim,
            img_shape=data_shape,
            output_dim=int(np.prod(data_shape))
        )
        self.discriminator = discriminator.Discriminator(img_shape=data_shape, output_dim=int(np.prod(data_shape)))

        self.validation_z = torch.rand(batch_size, self.hparams.latent_dim)

        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)

    def forward(self, z):
        return self.generator(z)

    @staticmethod
    def adversarial_loss(y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)

        # train generator
        if optimizer_idx == 0:
            # generate images
            self.generated_imgs = self(z)

            # ground truth result(ie: all fake)
            # put on GPU
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            tqdm_dict = {'g_loss': g_loss}
            output = discriminator.OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            writer.add_scalar('Generator Loss', g_loss, self.current_epoch)
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {'d_loss': d_loss}
            output = discriminator.OrderedDict({
                'loss': d_loss,
                'progress': tqdm_dict,
                'log': tqdm_dict
            })
            writer.add_scalar('Discriminator Loss', d_loss, self.current_epoch)
            return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_epoch_end(self) -> None:
        z = self.validation_z.type_as(self.generator.model[0].weight)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        # logger.experiment.add_image('generated_images', grid, self.current_epoch)
        writer.add_image('images', grid, global_step=self.current_epoch)
        writer.add_graph(self.discriminator, input_to_model=sample_imgs)
        writer.add_scalar('Lr', self.hparams.lr)
        # writer.add_hparams(self.hparams) # metric dict missing
        writer.close()


if __name__ == '__main__':
    size = 64
    dm = datamodule.DataModule()
    model = GAN(3, size, size)
    # logger = TensorBoardLogger('./tb_logs', name='CGAN')
    writer = SummaryWriter()
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=10,
        auto_lr_find='binary',
        auto_scale_batch_size='power',
        # precision=16,
        profiler='simple',
        # logger=logger,
    )
    trainer.fit(model, dm)
