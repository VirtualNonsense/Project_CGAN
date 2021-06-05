import torch


class Unpack(object):
    def __init__(self, a: int, b: int):
        self.__a = a
        self.__b = b

    def __call__(self, tensor: torch.tensor):
        return tensor.view(self.__a, self.__b)

    def __repr__(self):
        return self.__class__.__name__ + f"{self.__a}, {self.__b}"
