import numpy as np
from PIL import Image
from skimage.segmentation import find_boundaries
import os
from tqdm import trange
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300

data_dir = './data/raw/PANnuke/'  # location to extracted folds
output_dir = './data/interim/PANnuke/'  # location to save op data
unified_output_dir = './data/processed/PANnuke/'  # location to save op data

folds = os.listdir(data_dir)


def get_boundaries(raw_mask):
    '''
    for extracting instance boundaries form the goundtruth file
    '''
    bdr = np.zeros(shape=raw_mask.shape)
    for i in range(raw_mask.shape[-1] - 1):  # because last chnnel is background
        bdr[:, :, i] = find_boundaries(raw_mask[:, :, i], connectivity=1, mode='thick', background=0)
    bdr = np.sum(bdr, axis=-1)
    return bdr.astype(np.uint8)


for i, j in enumerate(folds):

    # get paths
    print('Loading Data for {}, Wait...'.format(j))
    img_path = data_dir + j + '/images/fold{}/images.npy'.format(i + 1)
    type_path = data_dir + j + '/images/fold{}/types.npy'.format(i + 1)
    mask_path = data_dir + j + '/masks/fold{}/masks.npy'.format(i + 1)
    print(40 * '=', '\n', j, 'Start\n', 40 * '=')

    # laod numpy files
    masks = np.load(file=mask_path, mmap_mode='r')  # read_only mode
    images = np.load(file=img_path, mmap_mode='r')  # read_only mode
    types = np.load(file=type_path)

    # creat directories to save images
    tissue_types = np.unique(types)
    for d in tissue_types:
        try:
            os.mkdir(output_dir + d)
            os.mkdir(output_dir + d + '/images')
            os.mkdir(output_dir + d + '/sem_masks')
            os.mkdir(output_dir + d + '/inst_masks')
        except FileExistsError:
            pass

    for k in trange(images.shape[0], desc='Writing files for {}'.format(j), total=images.shape[0]):
        raw_image = images[k, :, :, :].astype(np.uint8)
        raw_mask = masks[k, :, :, :]
        sem_mask = np.argmax(raw_mask, axis=-1).astype(np.uint8)
        # swaping channels 0 and 5 so that BG is at 0th channel
        sem_mask = np.where(sem_mask == 5, 6, sem_mask)
        sem_mask = np.where(sem_mask == 0, 5, sem_mask)
        sem_mask = np.where(sem_mask == 6, 0, sem_mask)

        tissue_type = types[k]
        instances = get_boundaries(raw_mask)

        raw_image_resized = Image.fromarray(raw_image).resize((128, 128))
        sem_mask_resized = Image.fromarray(sem_mask).resize((128, 128))
        instances_resized = Image.fromarray(instances).resize((128, 128))

        # save file in op dir
        sem_mask_resized.save(
            output_dir + '/{}/sem_masks/sem_{}_{}_{:05d}.png'.format(tissue_type, tissue_type, i + 1, k))
        instances_resized.save(
            output_dir + '/{}/inst_masks/inst_{}_{}_{:05d}.png'.format(tissue_type, tissue_type, i + 1, k))
        raw_image_resized.save(
            output_dir + '/{}/images/img_{}_{}_{:05d}.png'.format(tissue_type, tissue_type, i + 1, k))
        raw_image_resized.save(
            unified_output_dir + '/{}_{}_{:05d}.png'.format(tissue_type, i + 1, k))
