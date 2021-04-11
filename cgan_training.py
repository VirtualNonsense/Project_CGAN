import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optimizer
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vision_utils
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes as Axes
from os import environ
from typing import *
from PIL import ImageFile
import logging
from datetime import datetime, timedelta

import asyncio
from discriminator import Discriminator
from generator import Generator
from dataloader import MultiEpochsDataLoader


def _load_data_set(root_path: str, image_size: int, batch_size: int, load_worker: int,
                   shuffle: bool = True) -> DataLoader:
    data_set = datasets.ImageFolder(root=root_path, transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))

    return MultiEpochsDataLoader(data_set, batch_size=batch_size, shuffle=shuffle, num_workers=load_worker)


def _weights_init(m):
    classname = m.__class__.__name__
    # Discriminator
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    # Generator
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


async def _train(epochs: int,
                 dataloader: DataLoader,
                 discriminator: Discriminator,
                 generator: Generator,
                 generator_optimizer,
                 discriminator_optimizer,
                 generator_input_size: int,
                 device: torch.device,
                 criterion: nn.BCELoss,
                 fixed_noise: torch.Tensor,
                 ax: Axes,
                 real_label: Optional[int] = 1,
                 fake_label: Optional[int] = 0):
    # Lists to keep track of progress
    # img_list = []
    # generator_losses = []
    # discriminator_losses = []
    iterations = 0
    print("Starting Training Loop...")
    start = datetime.now()
    batches = len(dataloader)
    # For each epoch
    for epoch in range(epochs):
        # For each batch in the dataloader
        i = 0
        for data in dataloader:
            data: List[torch.Tensor] = data
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with all-real batch
            discriminator.zero_grad()
            # Format batch
            real_cpu: torch.Tensor = data[0].to(device)
            b_size = real_cpu.size()[0]
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output: torch.Tensor = discriminator(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, generator_input_size, 1, 1, device=device)
            # Generate fake image batch with G
            fake = generator(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = discriminator(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            discriminator_optimizer.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            generator_optimizer.step()
            # Output training stats
            if i == 0:
                elapsed_time = datetime.now() - start
                remaining_time = None
                if epoch > 0:
                    batches_done = i + 1 + epoch * batches
                    time_per_generation = elapsed_time.total_seconds() / batches_done
                    batches_left = (epochs - epoch) * batches - (i + 1)
                    remaining_time = timedelta(seconds=batches_left * time_per_generation)
                print(f"{elapsed_time} | {remaining_time} [{epoch + 1}/{epochs}]"
                      f"\tLoss_D: {errD.item():.4f}\t:Loss_G: {errG.item():.4f}"
                      f"\tD(x): {D_x:.4f}\tD(G(x)): {D_G_z1:.4f}/{D_G_z2:.4f}")


            # Save Losses for plotting later
            # generator_losses.append(errG.item())
            # discriminator_losses.append(errD.item())

            iterations += 1
            i += 1
        with torch.no_grad():
            fake = generator(fixed_noise).detach().cpu()
        # img_list.append(vision_utils.make_grid(fake, padding=2, normalize=True))
        ax.imshow(np.transpose(vision_utils.make_grid(fake, padding=2, normalize=True), (1, 2, 0)))
        ax.set_title(f"Epoch {epoch+1}/{epochs}")
        plt.pause(.01)
        await asyncio.sleep(.01)
        torch.save(generator, f"net_snapshot.pt")


async def _plot_update():
    while True:
        plt.pause(.1)
        await asyncio.sleep(.1)


async def _main():
    # logging.basicConfig(level=logging.DEBUG)
    # Set random seed for reproducibility
    manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)  # Root directory for dataset
    root_path = environ.get("CGAN_IMAGE_PATH")
    # root_path = r"S:\Users\Andre\Desktop\New folder"
    print(f"image path: {root_path}")
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    print(ImageFile.LOAD_TRUNCATED_IMAGES)

    # Number of workers for dataloader
    workers = 8

    # Batch size during training
    batch_size = 128

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 64

    amount_images = 64

    # Number of channels in the training images. For color images this is 3
    color_channel = 3

    # Size of z latent vector (i.e. size of generator input)
    generator_input_size = 100

    # Size of feature maps in generator
    generator_map_size = image_size

    # Size of feature maps in discriminator
    discriminator_map_size = image_size
    # Number of training epochs
    num_epochs = 1337

    # Learning rate for optimizers
    # learn_rate = 0.00075
    learn_rate = 0.0006

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    # Number of GPUs available. Use 0 for CPU mode.
    numb_gpu = 1

    # Decide which device we want to run on
    use_gpu = torch.cuda.is_available() and numb_gpu > 0
    device: torch.device = torch.device("cuda:0" if use_gpu else "cpu")

    print(f"Using {'GPU' if use_gpu else 'CPU'}")

    image_loader: DataLoader = _load_data_set(root_path=root_path,
                                              batch_size=batch_size,
                                              load_worker=workers,
                                              image_size=image_size)

    batch = next(iter(image_loader))
    fig = plt.figure()
    # ax0: Axes = fig.add_subplot(1, 2, 1)
    # ax0.set_title("training data")
    # ax0.imshow(np.transpose(vision_utils.make_grid(batch[0].to(device)[:amount_images], padding=2, normalize=True)
    #                         .cpu(), (1, 2, 0)))

    # Create the generator
    generator_net = Generator(numb_gpu,
                              feature_map_size=generator_map_size,
                              color_channels=color_channel,
                              input_size=generator_input_size).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (numb_gpu > 1):
        generator_net = nn.DataParallel(generator_net, list(range(numb_gpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    generator_net.apply(_weights_init)

    # Print the model
    print(generator_net)

    # Create the Discriminator
    discriminator_net = Discriminator(numb_gpu,
                                      feature_map_size=discriminator_map_size,
                                      color_channels=color_channel).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (numb_gpu > 1):
        discriminator_net = nn.DataParallel(discriminator_net, list(range(numb_gpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    discriminator_net.apply(_weights_init)

    # Print the model
    print(discriminator_net)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(amount_images, generator_input_size, 1, 1, device=device)

    # Establish convention for real and fake labels during training

    # Setup Adam optimizers for both G and D
    optimizerD = optimizer.Adam(discriminator_net.parameters(), lr=learn_rate, betas=(beta1, 0.999))
    optimizerG = optimizer.Adam(generator_net.parameters(), lr=learn_rate, betas=(beta1, 0.999))

    # Plot the fake images from the last epoch
    ax1: Axes = fig.add_subplot(1, 1, 1)
    plt.pause(.1)

    await asyncio.gather(_train(epochs=num_epochs,
                                dataloader=image_loader,
                                discriminator=discriminator_net,
                                generator=generator_net,
                                generator_optimizer=optimizerG,
                                discriminator_optimizer=optimizerD,
                                device=device,
                                generator_input_size=generator_input_size,
                                criterion=criterion,
                                ax=ax1,
                                fixed_noise=fixed_noise),
                         _plot_update())
    torch.save(generator_net, f"net_{num_epochs}.pt")
    print("trainig finished, model saved!")
    plt.show()


if __name__ == '__main__':
    loop = asyncio.get_event_loop()

    try:
        loop.run_until_complete(_main())
    except KeyboardInterrupt:
        print("user stop")
    finally:
        loop.close()
