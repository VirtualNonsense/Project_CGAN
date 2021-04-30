import torch
import torch.nn as nn
from typing import *


class GanGenerator(nn.Module):
    def __init__(self,
                 # number_of_gpus: int,
                 feature_map_size: int,
                 color_channels: int,
                 input_size: int,
                 num_classes: int,
                 img_size: int,
                 embed_size: int,
                 kernel_size: Union[int, Tuple[int, int]] = 4,
                 stride: Union[int, Tuple[int, int]] = 2,
                 padding: Union[int, Tuple[int, int]] = 1,
                 bias: bool = False):
        super(GanGenerator, self).__init__()
        self.img_size = img_size
        # self.number_of_gpus = number_of_gpus
        self.__input_size = input_size
        self.__kernel_size = (kernel_size, kernel_size) if type(kernel_size) is int else kernel_size
        self.__stride = (stride, stride) if type(stride) is int else stride
        self.__padding = (padding, padding) if type(padding) is int else padding
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # nn.ConvTranspose2d(input_size, feature_map_size * 16, kernel_size=self.__kernel_size, stride=self.__stride,
            #                    padding=(0, 0),
            #                    bias=bias),
            # nn.BatchNorm2d(feature_map_size * 16),
            # nn.ReLU(True),
            nn.ConvTranspose2d(input_size + embed_size, feature_map_size * 8, kernel_size=self.__kernel_size, stride=self.__stride,
                               padding=(0, 0),
                               bias=bias),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(feature_map_size * 8, feature_map_size * 4, kernel_size=self.__kernel_size,
                               stride=self.__stride,
                               padding=self.__padding, bias=bias),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, kernel_size=self.__kernel_size,
                               stride=self.__stride,
                               padding=self.__padding, bias=bias),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(feature_map_size * 2, feature_map_size, kernel_size=self.__kernel_size,
                               stride=self.__stride,
                               padding=self.__padding, bias=bias),
            nn.BatchNorm2d(feature_map_size),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(feature_map_size, color_channels, kernel_size=self.__kernel_size, stride=self.__stride,
                               padding=self.__padding,
                               bias=bias),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        self.embed = nn.Embedding(num_classes, embed_size)

    @property
    def input_size(self):
        return self.__input_size

    def forward(self, input_vector, labels):
        # latent vector z: N x noise_dim x 1 x 1
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        input_vector = torch.cat([input_vector, embedding], dim=1)
        return self.main(input_vector)
