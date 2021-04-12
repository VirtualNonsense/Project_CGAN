import random
from datetime import datetime, timedelta
from typing import *

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optimizer
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import ImageFile
from torch.utils.data import DataLoader

from dataloader import MultiEpochsDataLoader
from discriminator import Discriminator
from generator import Generator


def _load_data_set(set_root_path: str,
                   set_image_size: int,
                   set_batch_size: int,
                   load_worker: int,
                   shuffle: bool = True) -> DataLoader:
    data_set = datasets.ImageFolder(root=set_root_path, transform=transforms.Compose([
        transforms.Resize(set_image_size),
        transforms.CenterCrop(set_image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))

    return MultiEpochsDataLoader(data_set, batch_size=set_batch_size, shuffle=shuffle, num_workers=load_worker,
                                 pin_memory=True)


def _weights_init(m):
    classname = m.__class__.__name__
    # Discriminator
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    # Generator
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def _train(epochs: int,
           dataloader: DataLoader,
           discriminator: Discriminator,
           generator: Generator,
           generator_optimizer,
           discriminator_optimizer,
           generator_input_size: int,
           device: torch.device,
           criterion: nn.BCELoss,
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
            error_discriminator_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            error_discriminator_real.backward()
            d_x = output.mean().item()

            # Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, generator_input_size, 1, 1, device=device)
            # Generate fake image batch with G
            fake = generator(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = discriminator(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            error_discriminator_fake = criterion(output, label)
            # Calculate the gradients for this batch
            error_discriminator_fake.backward()
            # D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            error_discriminator = error_discriminator_real + error_discriminator_fake
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
            error_generator = criterion(output, label)
            # Calculate gradients for G
            error_generator.backward()
            d_g_z2 = output.mean().item()
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
                      f"\tLoss_D: {error_discriminator.item():.4f}\t:Loss_G: {error_generator.item():.4f}"
                      f"\tD(x): {d_x:.4f}\tD(G(x)): {d_g_z2:.4f}")
            iterations += 1
            i += 1


if __name__ == '__main__':
    manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)  # Root directory for dataset
    # root_path = environ.get("CGAN_IMAGE_PATH")
    # root_path = r"S:\Users\Andre\Desktop\New folder"
    root_path = r"C:\Users\Andre\Documents\New folder"
    print(f"image path: {root_path}")
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    print(ImageFile.LOAD_TRUNCATED_IMAGES)

    # Number of workers for dataloader
    workers = 8

    # Batch size during training
    batch_size = 4 * 128

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
    num_epochs = 10

    # Learning rate for optimizers
    # learn_rate = 0.00075
    learn_rate = 0.0006

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    # Number of GPUs available. Use 0 for CPU mode.
    numb_gpu = 1

    # Decide which device we want to run on
    use_gpu = torch.cuda.is_available() and numb_gpu > 0
    d: torch.device = torch.device("cuda:0" if use_gpu else "cpu")

    print(f"Using {'GPU' if use_gpu else 'CPU'}")

    image_loader: DataLoader = _load_data_set(set_root_path=root_path,
                                              set_batch_size=batch_size,
                                              load_worker=workers,
                                              set_image_size=image_size)

    batch = next(iter(image_loader))

    # Create the generator
    generator_net = Generator(numb_gpu,
                              feature_map_size=generator_map_size,
                              color_channels=color_channel,
                              input_size=gen_input_size).to(d)

    # Handle multi-gpu if desired
    if (d.type == 'cuda') and (numb_gpu > 1):
        generator_net = nn.DataParallel(generator_net, list(range(numb_gpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    generator_net.apply(_weights_init)

    # Print the model
    print(generator_net)

    # Create the Discriminator
    discriminator_net = Discriminator(numb_gpu,
                                      feature_map_size=discriminator_map_size,
                                      color_channels=color_channel).to(d)

    # Handle multi-gpu if desired
    if (d.type == 'cuda') and (numb_gpu > 1):
        discriminator_net = nn.DataParallel(discriminator_net, list(range(numb_gpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    discriminator_net.apply(_weights_init)

    # Print the model
    print(discriminator_net)

    # Initialize BCELoss function as criterion
    _criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(amount_images, gen_input_size, 1, 1, device=d)

    # Establish convention for real and fake labels during training

    # Setup Adam optimizers for both G and D
    optimizerD = optimizer.Adam(discriminator_net.parameters(), lr=learn_rate, betas=(beta1, 0.999))
    optimizerG = optimizer.Adam(generator_net.parameters(), lr=learn_rate, betas=(beta1, 0.999))

    # Plot the fake images from the last epoch
    loop_condition_container = [True]
    _train(epochs=num_epochs,
           dataloader=image_loader,
           discriminator=discriminator_net,
           generator=generator_net,
           generator_optimizer=optimizerG,
           discriminator_optimizer=optimizerD,
           device=d,
           generator_input_size=gen_input_size,
           criterion=_criterion)
    torch.save(generator_net, f"net_{num_epochs}.pt")
    print("trainig finished, model saved!")
