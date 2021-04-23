from PIL import Image, ImageFile
from PIL.PngImagePlugin import PngImageFile
from numba import njit
from typing import *
import glob
from os import environ, path


# @njit()
def resize(file_path, width: int, height: int):
    if path.isfile(file_path):
        im: PngImageFile = Image.open(file)
        if im.height != height or im.width != width:
            new_image = im.resize((width, height), Image.ANTIALIAS)
            new_image.save(file_path)


if __name__ == '__main__':
    size = 250
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    p = environ.get("CGAN_IMAGE_PATH")
    # p = r"S:\Users\Andre\Desktop\New folder"
    files = glob.glob(path.join(p, "**\\**\\*.png"), recursive=True)
    for i, file in enumerate(glob.glob(path.join(p, "**\\**\\*.png"), recursive=True)):
        print(f"{i/len(files) * 100:.2f}: {file}")
        resize(file, size, size)
