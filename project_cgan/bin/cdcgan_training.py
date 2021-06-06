import os

import pytorch_lightning as pl
import torch
from project_cgan.lib.c_dcgan import CDCGAN
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

if __name__ == "__main__":
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./checkpoints',
        save_top_k=3,
        mode='min'
    )
    color_channels = 3
    batch_size = 512 // 8

    image_size = 64
    label_dim = 4
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
    writer = SummaryWriter()
    model = CDCGAN(
        input_dim=latent_dim,
        amount_classes=label_dim,
        filter_sizes=num_filters,
        color_channels=color_channels,
        device=device,
        batch_size=batch_size,
        image_size=image_size
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