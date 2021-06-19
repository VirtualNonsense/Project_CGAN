import torch
# from project_cgan.lib.generator import DCGanGenerator
from project_cgan.lib.c_dcgan import Generator
import logging
import torchvision.utils as vision_utils
from matplotlib import pyplot as plt
import numpy as np
import torchvision

if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)
    path = r"gen_15901_256_100.pkl"
    batch_size = 1
    feature_map_size = 64
    color_channels = 3
    input_size = 100
    rows = 10
    num_filters = [1024, 512, 256, 128, 64, 32]
    amount_classes = 5
    # net = torch.load(path)
    use_gpu = torch.cuda.is_available()
    device: torch.device = torch.device("cpu")
    net = Generator(100, amount_classes, num_filters, 3, used_device=device)
    net.load_state_dict(torch.load(path))
    net.eval()
    # generator representation
    # each class has it's own 1 layer within this tensor.
    fixed = torch.tensor(
        [i % amount_classes for i in range(rows * amount_classes)],
        device=device, dtype=torch.long)

    noise = torch.randn(rows * amount_classes, 100, 1, 1, device=device)

    img = (net(noise, fixed).detach().cpu() + 1) / 2

    grid = torchvision.utils.make_grid(img, nrow=amount_classes)
    image = vision_utils.make_grid(grid, normalize=True)
    plt.figure()
    plt.imshow(np.transpose(image, (1, 2, 0)))
    plt.show()
