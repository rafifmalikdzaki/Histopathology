import os
import torch
from torch.utils.data import Dataset
import torchvision

class ImageDataset(Dataset):
    def __init__(self, image_dir):
        """
        Args:
            image_dir (string): Directory with all the images.
        """
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = torchvision.io.read_image(img_name)
        return image.float() / 255.0
