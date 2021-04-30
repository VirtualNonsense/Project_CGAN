"""
loosely based on
https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py#L68
"""
import os
import random
from datetime import datetime, timedelta
from typing import *
from os import environ, mkdir, path

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
        float_tensor,
        long_tensor,
        generator_input_size: int,
        classes: int,
        export_dir: str, sample_interval: int):
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(dataloader):

            batch_size = images.shape[0]

            # Adversarial ground truths
            valid = Variable(float_tensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(float_tensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(images.type(float_tensor))
            labels = Variable(labels.type(long_tensor))

            # -----------------
            #  Train Generator
            # -----------------

            generator_optimizer.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(float_tensor(np.random.normal(0, 1, (batch_size, generator_input_size))))
            gen_labels = Variable(long_tensor(np.random.randint(0, classes, batch_size)))

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
            validity_real = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            discriminator_optimizer.step()
            if i == 0:
                d_z = validity_real.view(-1).mean().item()
                d_g_z = validity_fake.view(-1).mean().item()
                print(
                    f"[Epoch {epoch+1}/{epochs}] "
                    f"[D loss: {d_loss.item():.3f}] "
                    f"[G loss: {g_loss.item():.3f}] "
                    f"[D(i): {d_z:.3f}] "
                    f"[D(G(z)): {d_g_z:.3f}] "
                )

        if epoch % sample_interval == 0:
            n_row = classes
            z = Variable(float_tensor(np.random.normal(0, 1, (n_row ** 2, generator_input_size))))
            labels = np.array([num for _ in range(n_row) for num in range(n_row)])
            labels = Variable(long_tensor(labels))
            with torch.no_grad():
                fake = generator(z, labels).detach().cpu()

            sample_image(n_row=10,
                         batches_done=epoch,
                         images=fake,
                         export_dir=export_dir)


if __name__ == '__main__':
    manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)  # Root directory for dataset
    # root_path = environ.get("CGAN_IMAGE_PATH")
    # root_path = r"S:\Users\Andre\Desktop\New folder"
    root_path = r"C:\Users\Andre\Documents\New folder"
    snapshot_directory = "../../snapshot/images"
    export_directory = "../../trained_models"
    print(f"image path: {root_path}")
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    print(ImageFile.LOAD_TRUNCATED_IMAGES)

    # Number of workers for dataloader
    workers = 8

    # Batch size during training
    batch_size = 4 * 128

    interval = 1

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 64

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
    learn_rate = 0.0001
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
    FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor

    # ----------
    #  Training
    # ----------

    if not path.isdir(export_directory):
        mkdir(export_directory)
    if not path.isdir(snapshot_directory):
        mkdir(snapshot_directory)
    _training(epochs=num_epochs,
              dataloader=image_loader,
              discriminator=discriminator,
              generator=generator,
              generator_optimizer=optimizer_G,
              discriminator_optimizer=optimizer_D,
              float_tensor=FloatTensor,
              long_tensor=LongTensor,
              generator_input_size=gen_input_size,
              export_dir=snapshot_directory,
              classes=classes,
              sample_interval=interval)
