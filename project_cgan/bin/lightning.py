"""
https://github.com/PyTorchLightning/pytorch-lightning
"""

import os

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import GPUStatsMonitor


class LitAutoEncoder(pl.LightningModule):

    def __init__(self, data_dir: str = './', batch_size: int = 128, num_workers: int = 12):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        # self.num_workers = num_workers
        self.num_workers = 12
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == '__main__':
    dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    train, val = random_split(dataset, [55000, 5000])

    # init model
    autoencoder = LitAutoEncoder(batch_size=256)

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs and more)
    trainer = pl.Trainer(
        gpus=1,
        auto_scale_batch_size='binsearch',
        auto_lr_find=True,
        # gpus=-1, # use all avaiable gpus
        log_gpu_memory='min_max',
        benchmark=True,
        limit_train_batches=0.1,  # only train 20% of an epoch
        profiler='simple',
        max_epochs=5,
        precision=16
    )
    # trainer = pl.Trainer()
    trainer.fit(autoencoder, DataLoader(train, num_workers=12, pin_memory=True), DataLoader(val, num_workers=12, pin_memory=True))
