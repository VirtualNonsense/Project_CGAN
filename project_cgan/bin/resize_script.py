from PIL import Image, ImageFile, ImageOps
from PIL.PngImagePlugin import PngImageFile
from numba import njit
from typing import *
import glob
from os import environ, path
from pathlib import Path


def change_aspect_ratio(image: Image, aspect_ratio: float = 1) -> Image:
    width, height = image.size
    old_aspect_ratio = width / height
    if old_aspect_ratio == aspect_ratio:
        return image
    new_im: Image
    if old_aspect_ratio < aspect_ratio:
        new_im = Image.new(image.mode, (height, int(height * aspect_ratio)), "white")
        x = (new_im.width - width) // 2
        new_im.paste(image, (x, 0))
        return new_im

    new_im = Image.new(image.mode, (int(width / aspect_ratio), width), "white")
    y = (new_im.height - height) // 2
    new_im.paste(image, (0, y))
    return new_im


def resize(file_path: Path, width: int, height: int, new_path: Optional[Path] = None, overwrite=True):
    if path.isfile(file_path):
        im: PngImageFile = Image.open(file)
        if im.height != height or im.width != width:
            p = new_path if new_path is not None else file_path
            if overwrite or not p.is_file():
                p.parents[0].mkdir(parents=True, exist_ok=True)
                new_image = change_aspect_ratio(im)
                new_image = new_image.resize((width, height), Image.ANTIALIAS)
                new_image.save(p)


if __name__ == '__main__':
    size = 256
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    p = r"S:\Users\Andre\Onedrive\CGAN\art\images"
    new_folder = r"S:\Users\Andre\Onedrive\CGAN\art\resized"
    files = glob.glob(path.join(p, "**\\*.jpg"), recursive=True)
    for i, file in enumerate(files):
        oldfile = Path(file)
        new_path = Path((new_folder if new_folder is not None else p) + "\\"
                        + oldfile.parents[0].name + "\\"
                        + oldfile.name)
        resize(oldfile, size, size, new_path)
