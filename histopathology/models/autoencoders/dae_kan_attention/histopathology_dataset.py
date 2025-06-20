import torch
import torchvision
import pandas as pd
from typing import Tuple, List, Optional
import os

def create_dataset(subset: str, image_path: Optional[str] = None) -> Tuple[List[str], torch.Tensor]:
    """
    Create a dataset by loading image paths and labels from a CSV file.
    
    Args:
        subset: Either 'train' or 'test' to specify which dataset to load
        image_path: Optional base path to the data directory. If None, uses default path.
    
    Returns:
        Tuple of (image_paths, labels) where image_paths is a list of absolute paths
        and labels is a tensor of corresponding labels
    """
    if image_path is None:
        image_path = "/home/dzakirm/Research/Histopathology/histopathology/data/processed"
    
    csv_path = os.path.join(image_path, f"{subset}.csv")
    print(f"Loading dataset from {csv_path}")
    
    df = pd.read_csv(csv_path)
    X = df['Image'].apply(lambda x: os.path.join(image_path, "HeparUnifiedPNG", x)).astype(str).to_list()
    y = torch.from_numpy(df.iloc[:, -1].to_numpy())
    return X, y


class ImageDataset(torch.utils.data.Dataset):
    """
    Dataset for loading histopathology images and their labels.
    
    Images are loaded on-demand and normalized to [0, 1] range.
    When using with DataLoader, set pin_memory=True for optimal GPU transfer.
    """
    def __init__(self, X: List[str], y: torch.Tensor, device: Optional[str] = None):
        """
        Args:
            X: List of image file paths
            y: Tensor of corresponding labels
            device: Optional device to place tensors on. If None, tensors remain on CPU
                   and can be moved to GPU via DataLoader's pin_memory
        """
        super().__init__()
        self.X = X
        self.y = y
        self.device = device

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load and normalize image to [0, 1]
        image = torchvision.io.read_image(self.X[i])
        image = image.type(torch.float32) / 255
        
        # Move to device if specified, otherwise leave on CPU for pin_memory
        if self.device is not None:
            image = image.to(self.device)
            label = self.y[i].to(self.device)
        else:
            label = self.y[i]
            
        return image, label

    def __len__(self) -> int:
        return len(self.X)

#%%
