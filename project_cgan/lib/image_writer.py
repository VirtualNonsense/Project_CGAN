import os

import PIL.Image
import imageio
import pathlib
import torch
import torchvision
from typing import *


class ImageWriter:
    def __init__(self, save_dir: [str, pathlib.Path], file_prefix: str, expected_digits: int = 4,
                 file_ext: Optional[str] = "png"):
        self.file_ext = file_ext
        self.expected_digits = expected_digits
        if isinstance(save_dir, str):
            save_dir = pathlib.Path(save_dir)
        self.save_dir = save_dir
        self.file_prefix = file_prefix
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.image_enumerator = 0

    def add_image(self, tensor: Union[torch.Tensor, PIL.Image.Image], comment: Optional[str] = None) -> pathlib.Path:
        if isinstance(tensor, torch.Tensor):
            tensor = torchvision.transforms.ToPILImage()(tensor.detach().cpu())
        image: PIL.Image.Image = tensor
        comment = "" if comment is None else comment
        p = self.save_dir.joinpath(
            f"{str(self.image_enumerator).zfill(self.expected_digits)}{self.file_prefix}_{comment}.{self.file_ext}")
        image.save(p)
        self.image_enumerator += 1
        return p


class GifWriter(ImageWriter):
    def __init__(self, save_dir: [str, pathlib.Path], file_prefix: str, gif_name: str, fps: int = 5,
                 delete_images: bool = False, file_ext: Optional[str] = "png", expected_digits: int = 4):
        super().__init__(save_dir=save_dir, file_prefix=file_prefix, file_ext=file_ext, expected_digits=expected_digits)
        self.fps = fps
        self.delete_images = delete_images
        self.gif_name = gif_name
        self.paths = []

    def add_image(self, tensor: Union[torch.Tensor, PIL.Image.Image], comment: Optional[str] = None) -> pathlib.Path:
        p = super().add_image(tensor, comment)
        self.paths.append(p)
        return p

    def save_to_gif(self):
        images = []
        for p in self.paths:
            images.append(imageio.imread(p))
        path = self.save_dir.joinpath(f"{str(self.image_enumerator).zfill(self.expected_digits)}{self.file_prefix}.gif")
        imageio.mimsave(path, images, fps=self.fps)
        if self.delete_images:
            for p in self.paths:
                os.remove(p)
