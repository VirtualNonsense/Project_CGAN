import os
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import generator
import discriminator
import datamodule


class GAN(pl.LightningModule):
    def __init__(self,
                 channels, width, heigth, num_classes: int, embed_size: int, latent_dim: int = 100, lr: float = 0.002, b1: float = 0.5, b2: float = 0.999,
                 batch_size: int = 1024, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # networks
        data_shape = (channels, width, heigth)
        self.generator = generator.Generator(
            num_classes=num_classes,
            embed_size=embed_size,
            latent_dim=self.hparams.latent_dim,
            img_shape=data_shape,
            output_dim=int(np.prod(data_shape)
                           )
        )
        self.discriminator = discriminator.Discriminator(
            img_shape=data_shape,
            output_dim=int(np.prod(data_shape)),
            num_classes=num_classes,
        )

        self.validation_z = torch.rand(batch_size, self.hparams.latent_dim)

        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)

    def forward(self, z, labels):
        return self.generator(z, labels)

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
        z = self.validation_z.type_as(self.generator.model[0].weight)[:64]

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        writer.add_image('images', grid, global_step=self.current_epoch)
        writer.add_graph(self.discriminator, input_to_model=sample_imgs)
        # writer.add_graph(self.generator, input_to_model=sample_imgs)
        # writer.add_scalar('Lr', self.hparams.lr)
        # writer.add_hparams({
        #     'lr': self.hparams.lr,
        #     'bsize': self.batch_size,
        # })
        writer.close()

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self.discriminator(x)
    #     g_loss = F.cross_entropy(y_hat, y)
    #     writer.add_scalar('g_loss', g_loss)


if __name__ == '__main__':
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./checkpoints',
        save_top_k=3,
        mode='min'
    )
    size = 64
    num_classes = 13
    embed_size = 100
    data_dir = os.environ['CGAN_SORTED']
    dm = datamodule.DataModule(data_dir)
    model = GAN(3, size, size, num_classes, embed_size)
    # logger = TensorBoardLogger('./tb_logs', name='CGAN')
    writer = SummaryWriter()
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=50,
        # auto_lr_find=True,
        # auto_scale_batch_size=True,
        # precision=16,
        profiler='simple',
        callbacks=[checkpoint_callback]
    )
    trainer.tune(model, datamodule=dm)
    trainer.fit(model, dm)
    # lr_finder = trainer.tuner.lr_find(model)
    # fig = lr_finder.plot(suggest=True)
    # fig.show()
