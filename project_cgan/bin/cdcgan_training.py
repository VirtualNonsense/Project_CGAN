import os
import torch
import pytorch_lightning as pl
from project_cgan.lib.c_dcgan import CDCGAN
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
from project_cgan.lib.image_writer import GifWriter
from pytorch_lightning.callbacks import ModelCheckpoint
from project_cgan.lib.dataloader import MultiEpochsDataLoader

if __name__ == "__main__":
    name = "cover_art"
    misc = ""
    color_channels = 3
    batch_size = 128

    image_size = 128
    latent_dim = 100
    num_filters = [1024, 512, 256, 128, 64]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path = os.environ['CGAN_SORTED']
    # path = r'C:\Users\Andre\Documents\resized_celebA_hair_sorted'
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
    run_tag = f"{name}_{misc}_{image_size}_{latent_dim}_{filter_label}"
    writer = SummaryWriter(comment=run_tag)
    image_writer = GifWriter(save_dir="/images", file_prefix=f"cdcgan_{run_tag}", gif_name=f"cdcgan_{run_tag}")
    model = CDCGAN(
        input_dim=latent_dim,
        amount_classes=label_dim,
        filter_sizes=num_filters,
        color_channels=color_channels,
        device=device,
        batch_size=batch_size,
        image_size=image_size,
        writer=writer,
        image_writer=image_writer,
        image_intervall=1
    )
    opt_checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints/cdcgan/',
        save_top_k=2,
        monitor="g_loss",
        filename=run_tag + '-{epoch:02d}-{g_loss:.2f}',
        mode='min'
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints/cdcgan/',
        period=10,
        filename=run_tag + '-{epoch:02d}',
    )

    trainer = pl.Trainer(
        max_epochs=500,
        gpus=1 if torch.cuda.is_available() else 0,
        progress_bar_refresh_rate=1,
        profiler='simple',
        callbacks=[
            opt_checkpoint_callback,
            checkpoint_callback
        ],
    )
    # trainer.tune(model, dm)
    trainer.fit(model, dataloader)
    image_writer.save_to_gif()
