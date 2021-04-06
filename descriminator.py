import torch.nn as nn
from typing import *


class Discriminator(nn.Module):
    def __init__(self, number_of_gpus: int, feature_map_size: int, color_channels: int,
                 kernel_size: Union[int, Tuple[int, int]] = 4, stride: Union[int, Tuple[int, int]] = 2,
                 padding: Union[int, Tuple[int, int]] = 1, bias: bool = False, inplace: bool = True):
        super(Discriminator, self).__init__()
        self.__kernel_size = (kernel_size, kernel_size) if type(kernel_size) is int else kernel_size
        self.__stride = (stride, stride) if type(stride) is int else stride
        self.__padding = (padding, padding) if type(padding) is int else padding
        self.number_of_gpus = number_of_gpus
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(color_channels, feature_map_size, kernel_size=self.__kernel_size,
                      stride=self.__stride,
                      padding=self.__padding, bias=bias),
            nn.LeakyReLU(0.2, inplace=inplace),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(feature_map_size, feature_map_size * 2, kernel_size=self.__kernel_size,
                      stride=self.__stride,
                      padding=self.__padding, bias=bias),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.LeakyReLU(0.2, inplace=inplace),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(feature_map_size * 2, feature_map_size * 4, kernel_size=self.__kernel_size,
                      stride=self.__stride,
                      padding=self.__padding, bias=bias),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.LeakyReLU(0.2, inplace=inplace),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(feature_map_size * 4, feature_map_size * 8, kernel_size=self.__kernel_size,
                      stride=self.__stride,
                      padding=self.__padding, bias=bias),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.LeakyReLU(0.2, inplace=inplace),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(feature_map_size * 8, 1, kernel_size=self.__kernel_size,
                      stride=self.__stride,
                      padding=(0, 0), bias=inplace),
            nn.Sigmoid()
        )

    def forward(self, input_vector):
        return self.main(input_vector)
