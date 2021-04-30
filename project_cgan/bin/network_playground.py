import torch
from project_cgan.lib.generator import GanGenerator
import logging
import torchvision.utils as vision_utils
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    path = r"g_net_500.pt"
    batch_size = 64
    feature_map_size = 64
    color_channels = 3
    input_size = 100
    net = GanGenerator(feature_map_size=feature_map_size,
                       color_channels=color_channels,
                       input_size=input_size)
    net.load_state_dict(torch.load(path))
    net.eval()
    # net = torch.load(path)
    use_gpu = torch.cuda.is_available()
    device: torch.device = torch.device("cpu")

    noise = torch.randn(batch_size, 100, 1, 1, device=device)
    image = vision_utils.make_grid(net(noise).detach().cpu(), normalize=True)
    plt.figure()
    plt.imshow(np.transpose(image, (1, 2, 0)))
    plt.show()
