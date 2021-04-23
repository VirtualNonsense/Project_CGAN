import torch
from project_cgan.lib.generator import Generator
import logging
import torchvision.utils as vision_utils
from matplotlib import pyplot as plt
import numpy as np
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    batch_size = 64
    input_size = 100
    net: Generator = torch.load("net.pt")
    use_gpu = torch.cuda.is_available()
    device: torch.device = torch.device("cuda:0" if use_gpu else "cpu")

    noise = torch.randn(batch_size, 100, 1, 1, device=device)
    image = vision_utils.make_grid(net(noise).detach().cpu(), normalize=True)
    plt.figure()
    plt.imshow(np.transpose(image, (1, 2, 0)))
    plt.show()
