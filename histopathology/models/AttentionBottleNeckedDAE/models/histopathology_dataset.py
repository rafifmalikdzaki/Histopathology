import torch
import torchvision
import pandas as pd

import numpy as np
import os
import cv2

def create_dataset(subset: str, image_path: str = None):
    if image_path is None:
        image_path = f"../../../data/processed/{subset}.csv"
    df = pd.read_csv(image_path)
    X = df['Image'].apply(lambda x: "../../../data/processed/HeparUnifiedPNG/" + x).astype(str).to_list()
    y = torch.from_numpy(df.iloc[:, -1].to_numpy())
    return X, y


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, X: torch.tensor, y: torch.tensor, device='cpu'):
        super().__init__()

        self.device = device

        self.X = X
        self.y = y

    def __getitem__(self, i: int):
        image = torchvision.io.read_image(self.X[i])
        image = image.type(torch.float32) / 255
        return image.to(self.device), self.y[i].to(self.device)

    def __len__(self):
        return len(self.X)

#%%
