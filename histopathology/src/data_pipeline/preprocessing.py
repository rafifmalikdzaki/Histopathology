import numpy as np
from tiatoolbox.tools.stainnorm import VahadaneNormalizer
from multiprocessing import Pool

import os
import glob
import numpy as np
from patchify import patchify, unpatchify
from PIL import Image
import tifffile


def normalization(preProc_path, postProc_path, normalizer):
    tilepath, tilename = os.path.split(preProc_path)
    tile_path = os.path.join(tilepath, tilename)

    image = tifffile.imread(tile_path)
    norm_image = normalizer.transform(image)

    tifffile.imwrite(f"{postProc_path}/{tilename}", norm_image)


def parallelNormalization(preProc_path, postProc_path, file_groups):

    for g in file_groups:

        norm = VahadaneNormalizer()

        path_g = os.path.join(preProc_path, g)
        post_tile_g = os.path.join(postProc_path, g)

        # print(path_patch_g, path_g)
        tile_paths = glob.glob(os.path.join(
            path_g, '*.tiff'))  # Get all .tif files

        image_arr = tifffile.imread(tile_paths[0])
        norm.fit(image_arr)

        with Pool() as pool:  # Create a multiprocessing pool
            pool.starmap(normalization, [(tile_path, post_tile_g, norm)
                                         for tile_path in tile_paths])


if __name__ == "__main__":
    path = "../../data/interim/tiles/"
    tile_path = "../../data/interim/tiles_preproc/"

    parallelNormalization(path, tile_path, [
        "HFD10X", "HFD20X", "HFD40X", "ND200X", "ND400X"])
