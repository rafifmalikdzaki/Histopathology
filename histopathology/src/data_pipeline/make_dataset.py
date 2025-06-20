import numpy as np
from tiatoolbox.tools import stainnorm
from multiprocessing import Pool
from .histopathology_dataset import ImageDataset

import os
import glob
import numpy as np
from patchify import patchify, unpatchify
from PIL import Image


def generate_patches(slidepath, patch_path, patch_size=128):
    slidepath, slidename = os.path.split(slidepath)
    slide_path = os.path.join(slidepath, slidename)
    # print(slide_path)

    base_name = os.path.splitext(slidename)[0]
    # patch_name = f"Tiles_{base_name}"
    patch_path = os.path.join(patch_path)
    # os.makedirs(patch_path)

    image = Image.open(slide_path)
    image = np.asarray(image)

    image_h, image_w, channel_n = image.shape
    patch_height, patch_width, stride = patch_size, patch_size, patch_size
    patch_shape = (patch_height, patch_width, 3)

    patches = patchify(image, patch_shape, step=stride)

    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patch = patches[i, j, 0]
            patch = Image.fromarray(patch)
            patch.save(f"{patch_path}/{base_name}_T{i}{j}.tiff")


def parallelPatching(path, patch_path, file_groups):

    for g in file_groups:

        path_g = os.path.join(path, g)
        path_patch_g = os.path.join(patch_path, g)
        # print(path_patch_g, path_g)
        slide_paths = glob.glob(os.path.join(
            path_g, '*.tif'))  # Get all .tif files

        with Pool() as pool:  # Create a multiprocessing pool
            pool.starmap(generate_patches, [(slide_path, path_patch_g)
                                            for slide_path in slide_paths])


def create_dataset(split: str):
    """Create train/test dataset"""
    base_dir = './data/processed/HeparUnifiedPNG/'
    return ImageDataset(image_dir=os.path.join(base_dir, split))


if __name__ == "__main__":
    path = "../../data/interim/slides/"
    patch_path = "../../data/interim/tiles/"

    parallelPatching(path, patch_path, [
                     "HFD10X", "HFD20X", "HFD40X", "ND200X", "ND400X"])
