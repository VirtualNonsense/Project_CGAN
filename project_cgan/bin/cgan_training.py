"""
loosely based on
https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py#L68
"""
import os
import random
from datetime import datetime, timedelta
from typing import *
from os import environ, mkdir, path
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optimizer
from torch.autograd import Variable
from torchvision.utils import save_image
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import ImageFile
from torch.utils.data import DataLoader

from project_cgan.lib.generator import CGanGenerator
from project_cgan.lib.discriminator import CGanDiscriminator
from project_cgan.lib.dataloader import MultiEpochsDataLoader


def _load_data_set(set_root_path: str,
                   set_image_size: int,
                   set_batch_size: int,
                   load_worker: int,
                   shuffle: bool = True) -> DataLoader:
    data_set = datasets.ImageFolder(root=set_root_path,
                                    transform=transforms.Compose([
                                        transforms.Resize(set_image_size),
                                        transforms.CenterCrop(set_image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ]))

    return MultiEpochsDataLoader(data_set, batch_size=set_batch_size, shuffle=shuffle, num_workers=load_worker,
                                 pin_memory=True)


def sample_image(n_row,
                 batches_done,
                 images,
                 export_dir):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    save_image(images.data, os.path.join(export_dir, f"{batches_done}.png"), nrow=n_row, normalize=True)


def _training(
        epochs: int,
        dataloader: DataLoader,
        discriminator: CGanDiscriminator,
        generator: CGanGenerator,
        generator_optimizer,
        discriminator_optimizer,
        device: torch.device,
        generator_input_size: int,
        classes: int,
        image_export_dir: str,
        snapshot_dir: str,
        sample_interval: int):
    batches = len(dataloader)
    loop_start = datetime.now()

    loc_scale = (-1, 1)
    image_rows = 10
    fixed_noise = torch.tensor(
        np.random.normal(loc_scale[0], loc_scale[1],
                         (image_rows * classes, generator_input_size)),
        device=device, dtype=torch.float)
    fixed_labels = torch.tensor(
        [i % classes for i in range(image_rows * classes)],
        device=device, dtype=torch.long)
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(dataloader):
            images = torch.tensor(images, device=device)
            labels = torch.tensor(labels, device=device)
            batch_size = images.shape[0]

            # Adversarial ground truths
            valid = torch.ones((batch_size, 1), dtype=torch.float, device=device)
            fake = torch.zeros((batch_size, 1), dtype=torch.float, device=device)

            # -----------------
            #  Train Generator
            # -----------------

            generator_optimizer.zero_grad()

            # Sample noise and labels as generator input
            z = torch.randn((batch_size, generator_input_size), device=device, dtype=torch.float)
            gen_labels = torch.tensor(np.random.randint(0, classes, batch_size), dtype=torch.long, device=device)

            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            generator_optimizer.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            discriminator_optimizer.zero_grad()

            # Loss for real images
            validity_real = discriminator(images, labels)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss)

            d_loss.backward()
            discriminator_optimizer.step()
            if i == 0:
                c = datetime.now()
                elapsed_time = c - loop_start
                d_i = validity_real.view(-1).mean().item()
                d_g_z = validity_fake.view(-1).mean().item()
                batches_done = i + 1 + epoch * batches
                time_per_generation = elapsed_time.total_seconds() / batches_done
                batches_left = (epochs - epoch) * batches - (i + 1)
                remaining_time = timedelta(seconds=batches_left * time_per_generation)
                print(f"{c} | {elapsed_time} | {remaining_time} [{epoch + 1}/{epochs}]"
                      f"\tLoss_D: {d_loss.item():.4f}\t:Loss_G: {g_loss.item():.4f}"
                      f"\tD(i): {d_i:.4f}\tD(G(z)): {d_g_z:.4f}")
        if epoch % sample_interval == 0:

            with torch.no_grad():
                fake = generator(fixed_noise, fixed_labels).detach().cpu()

            sample_image(n_row=classes,
                         batches_done=epoch,
                         images=fake,
                         export_dir=image_export_dir)
            torch.save(generator.state_dict(), path.join(snapshot_dir, "cgan_g_snapshot.pt"))
            torch.save(discriminator.state_dict(), path.join(snapshot_dir, "cgan_d_snapshot.pt"))
            torch.save(optimizer_D.state_dict(), path.join(snapshot_dir, "cgan_optD_snapshot.pt"))
            torch.save(optimizer_G.state_dict(), path.join(snapshot_dir, "cgan_optG_snapshot.pt"))


if __name__ == '__main__':
    start_time = datetime.now()
    manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)  # Root directory for dataset
    # root_path = environ.get("CGAN_SORTED")
    # root_path = r"S:\Users\Andre\Desktop\New folder"
    root_path = r"S:\Users\Andre\Repositories\Python\Project_CGAN\test"
    image_directory = "../../snapshot/images/"
    snapshot_directory = "../../snapshot/"
    export_directory = "../../trained_models/"
    print(f"image path: {root_path}")
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Number of workers for dataloader
    workers = 8

    # Batch size during training
    batch_size = 128

    interval = 1

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 128

    amount_images = 64

    # Number of channels in the training images. For color images this is 3
    color_channel = 3

    # Size of z latent vector (i.e. size of generator input)
    gen_input_size = 100

    # Size of feature maps in generator
    generator_map_size = image_size

    # Size of feature maps in discriminator
    discriminator_map_size = image_size
    # Number of training epochs
    num_epochs = 1000

    # Learning rate for optimizers
    # learn_rate = 0.0006
    learn_rate = 0.0002
    epsilon = 1e-8

    # Beta1 hyper parameter for Adam optimizers
    # beta1 = 0.5
    beta1 = 0.9
    beta2 = 0.999

    # Number of GPUs available. Use 0 for CPU mode.
    numb_gpu = 1

    # Decide which device we want to run on
    use_gpu = torch.cuda.is_available() and numb_gpu > 0
    d: torch.device = torch.device("cuda:0" if use_gpu else "cpu")
    # Configure data loader
    image_loader: DataLoader = _load_data_set(set_root_path=root_path,
                                              set_batch_size=batch_size,
                                              load_worker=workers,
                                              set_image_size=image_size)

    # that should work
    # noinspection PyUnresolvedReferences
    classes: int = len(image_loader.dataset.classes)
    print(f"classes: {classes}")
    print(f"Using {'GPU' if use_gpu else 'CPU'}")

    # Loss functions
    adversarial_loss = torch.nn.MSELoss()

    # Initialize generator and discriminator
    generator = CGanGenerator(classes=classes,
                              latent_dim=gen_input_size,
                              img_shape=(color_channel, image_size, image_size))
    discriminator = CGanDiscriminator(classes=classes,
                                      img_shape=(color_channel, image_size, image_size))

    if use_gpu:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=learn_rate, betas=(beta1, beta2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learn_rate, betas=(beta1, beta2))
    #
    # ----------
    #  Training
    # ----------
    p = Path(export_directory)
    p.mkdir(parents=True, exist_ok=True)
    p = Path(image_directory)
    p.mkdir(parents=True, exist_ok=True)
    p = Path(snapshot_directory)
    p.mkdir(parents=True, exist_ok=True)
    _training(epochs=num_epochs,
              dataloader=image_loader,
              discriminator=discriminator,
              generator=generator,
              generator_optimizer=optimizer_G,
              discriminator_optimizer=optimizer_D,
              device=device,
              generator_input_size=gen_input_size,
              image_export_dir=image_directory,
              classes=classes,
              sample_interval=interval,
              snapshot_dir=snapshot_directory)
    print("saving generator...")
    torch.save(generator.state_dict(), path.join(export_directory, f"cgan{num_epochs}.pt"))
    print(f"{datetime.now() }Training finished!\nTotal training time:{datetime.now() - start_time}")
