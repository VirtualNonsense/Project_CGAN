import torch
import torch.nn as nn
from typing import *


def __gen_block(input_size, output_size, kernel, stride, padding, negative_slope: float,
                inplace: bool, bias: bool, batch_norm: bool = True):
    if batch_norm:
        return [
            nn.Conv2d(input_size, output_size, kernel_size=kernel, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(output_size),
            nn.LeakyReLU(negative_slope, inplace=inplace)
        ]

    return [
        nn.Conv2d(input_size, output_size, kernel_size=kernel, stride=stride, padding=padding, bias=bias),
        nn.LeakyReLU(negative_slope, inplace=inplace)
    ]


def _gen_layers(layer_count: int, input_layer_size, output_layer_size, map_size, kernel, padding, stride,
                inplace: bool, bias: bool, negative_slope: float) -> List[any]:
    layers = __gen_block(input_layer_size, map_size, kernel=kernel, stride=stride, padding=padding,
                         inplace=inplace, negative_slope=negative_slope, bias=bias,
                         batch_norm=False)
    new_map_size = map_size * 2
    if layer_count > 2:
        for i in range(layer_count - 2):
            layers += __gen_block(map_size, new_map_size, kernel=kernel, stride=stride,
                                  padding=padding,
                                  inplace=inplace, negative_slope=negative_slope, bias=bias)
            map_size = new_map_size
            new_map_size *= 2
    layers += [
        nn.Conv2d(map_size, output_layer_size, kernel_size=kernel, stride=(1, 1), padding=(0, 0), bias=bias),
        nn.Sigmoid()
    ]
    return layers


class GanDiscriminator(nn.Module):
    def __init__(self,
                 # number_of_gpus: int,
                 feature_map_size: int,
                 input_channels: int,
                 kernel_size: Union[int, Tuple[int, int]] = 4,
                 stride: Union[int, Tuple[int, int]] = 2,
                 padding: Union[int, Tuple[int, int]] = 1,
                 bias: bool = False,
                 inplace: bool = True,
                 negative_slope: float = 0.2):
        super(GanDiscriminator, self).__init__()
        self.__kernel_size = (kernel_size, kernel_size) if type(kernel_size) is int else kernel_size
        self.__stride = (stride, stride) if type(stride) is int else stride
        self.__padding = (padding, padding) if type(padding) is int else padding
        # self.number_of_gpus = number_of_gpus
        self.main = nn.Sequential(
            *_gen_layers(5,
                         input_channels,
                         1,
                         feature_map_size,
                         self.__kernel_size,
                         self.__padding,
                         self.__stride,
                         bias=bias,
                         inplace=inplace,
                         negative_slope=negative_slope)
        )

    def forward(self, input_vector) -> torch.Tensor:
        return self.main(input_vector)


class CganDiscriminator(GanDiscriminator):
    r"""
    roughly oriented on
    https://github.com/Lornatang/CGAN-PyTorch/blob/master/cgan_pytorch/models.py
    """
    def __init__(self,
                 feature_map_size: int,
                 input_channels: int,
                 num_classes: int,
                 kernel_size: Union[int, Tuple[int, int]] = 4,
                 stride: Union[int, Tuple[int, int]] = 2,
                 padding: Union[int, Tuple[int, int]] = 1,
                 bias: bool = False,
                 inplace: bool = True,
                 negative_slope: float = 0.2):
        super().__init__(
                 feature_map_size=feature_map_size,
                 input_channels=input_channels + num_classes,
                 kernel_size=kernel_size,
                 stride=stride,
                 padding=padding,
                 bias=bias,
                 inplace=inplace,
                 negative_slope=negative_slope)
        self.label_embedding = nn.Embedding(num_classes, num_classes)

    def forward(self, input_vector, labels: list = None) -> torch.Tensor:

        flattened_input = torch.flatten(input_vector, 1)
        conditional = self.label_embedding(labels)
        conditional_input = torch.cat([flattened_input, conditional], dim=-1)
        output = self.main(conditional_input)
        return output
