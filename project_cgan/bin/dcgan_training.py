import os

import pytorch_lightning as pl
import torch
from project_cgan.lib.dcgan import DCGAN
from project_cgan.lib.dataloader import MultiEpochsDataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

if __name__ == "__main__":
    name = "cover_art"
    misc = ""
    color_channels = 3
    batch_size = 128

    image_size = 256
    latent_dim = 100
    num_filters = [1024, 512, 256, 128, 64, 32]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path = os.environ['CGAN_IMAGE_PATH']
    print(f"grabbing trainingsdata from: {path}")

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    data = datasets.ImageFolder(root=path, transform=transform)

    dataloader = MultiEpochsDataLoader(data, batch_size=batch_size, shuffle=True, num_workers=6)
    label_dim = len(dataloader.dataset.classes)
    filter_label = "-".join([str(i) for i in num_filters])
    run_tag = f"dcgan_{name}_{misc}_{image_size}_{latent_dim}_{filter_label}"
    writer = SummaryWriter(comment=run_tag)
    model = DCGAN(
        input_dim=latent_dim,
        filter_sizes=num_filters,
        color_channels=color_channels,
        device=device,
        batch_size=batch_size,
        image_size=image_size,
        writer=writer
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints',
        save_top_k=5,
        monitor="g_loss",
        filename=run_tag + '-{epoch:02d}-{g_loss:.2f}',
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=50000,
        gpus=1 if torch.cuda.is_available() else 0,
        progress_bar_refresh_rate=1,
        profiler='simple',
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, dataloader)
