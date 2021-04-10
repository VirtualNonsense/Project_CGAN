import torch.nn as nn
from typing import *


class Discriminator(nn.Module):
    def __init__(self,
                 number_of_gpus: int,
                 feature_map_size: int,
                 color_channels: int,
                 kernel_size: Union[int, Tuple[int, int]] = 4,
                 stride: Union[int, Tuple[int, int]] = 2,
                 padding: Union[int, Tuple[int, int]] = 1,
                 bias: bool = False,
                 inplace: bool = True,
                 negative_slope: float = 0.2):
        super(Discriminator, self).__init__()
        self.__kernel_size = (kernel_size, kernel_size) if type(kernel_size) is int else kernel_size
        self.__stride = (stride, stride) if type(stride) is int else stride
        self.__padding = (padding, padding) if type(padding) is int else padding
        self.number_of_gpus = number_of_gpus
        self.main = nn.Sequential(
            *Discriminator.__gen_layers(6,
                                        color_channels,
                                        1,
                                        feature_map_size,
                                        self.__kernel_size,
                                        self.__padding,
                                        self.__stride,
                                        bias=bias,
                                        inplace=inplace,
                                        negative_slope=negative_slope)
        )

    @staticmethod
    def __gen_block(input_size, output_size, kernel, stride, padding, negative_slope: float,
                    inplace: bool, bias: bool):
        return [
            nn.Conv2d(input_size, output_size, kernel_size=kernel, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(output_size),
            nn.LeakyReLU(negative_slope, inplace=inplace)
        ]

    @staticmethod
    def __gen_layers(layer_count: int, input_layer_size, output_layer_size, map_size, kernel, padding, stride,
                     inplace: bool, bias: bool, negative_slope: float) -> List[
        any]:
        layers = Discriminator.__gen_block(input_layer_size, map_size, kernel=kernel, stride=stride, padding=padding,
                                           inplace=inplace, negative_slope=negative_slope, bias=bias)
        new_map_size = map_size * 2
        if layer_count > 2:
            for i in range(layer_count - 2):
                layers += Discriminator.__gen_block(map_size, new_map_size, kernel=kernel, stride=stride,
                                                    padding=padding,
                                                    inplace=inplace, negative_slope=negative_slope, bias=bias)
                map_size = new_map_size
                new_map_size *= 2
        layers += Discriminator.__gen_block(map_size, output_layer_size, kernel=kernel, stride=stride, padding=(0, 0),
                                            inplace=inplace, negative_slope=negative_slope, bias=bias)
        layers += [nn.Sigmoid()]
        return layers

    def forward(self, input_vector):
        return self.main(input_vector)
