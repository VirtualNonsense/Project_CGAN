import torch
import pytorch_lightning as pl
from project_cgan.lib.cgan import CGAN
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.callbacks import ModelCheckpoint
from project_cgan.lib.dataloader import MultiEpochsDataLoader

if __name__ == "__main__":
    name = "cover_art"
    misc = ""
    color_channels = 3
    batch_size = 512

    image_size = 64
    latent_dim = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path = os.environ['CGAN_SORTED']
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
    run_tag = f"{name}_{misc}_{image_size}_{latent_dim}"
    writer = SummaryWriter(comment=run_tag)
    model = CGAN(
        input_dim=latent_dim,
        amount_classes=label_dim,
        color_channels=color_channels,
        device=device,
        batch_size=batch_size,
        image_size=image_size,
    )
    model.writer = writer
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints',
        save_top_k=5,
        monitor="g_loss",
        filename='sample-' + name + '-{epoch:02d}-{g_loss:.2f}',
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=50000,
        gpus=1 if torch.cuda.is_available() else 0,
        progress_bar_refresh_rate=1,
        profiler='simple',
        callbacks=[checkpoint_callback],
    )
    # trainer.tune(model, dm)
    trainer.fit(model, dataloader)
