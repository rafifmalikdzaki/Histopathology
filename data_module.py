from __future__ import annotations

"""Lightning-friendly replacement for the old ``load_dataset.py``
    ----------------------------------------------------------------

▪ Keeps *exactly* the same CSV / 🤗 Hub workflow (local TSV or ``datasets.load_dataset``)
▪ Re-uses the MONAI transform pipeline you already tuned
▪ Exposes a `RetinaDataModule` so the training script can simply do::

    dm = RetinaDataModule(cfg)
    dm.setup()
    trainer.fit(model, datamodule=dm)

▪ Still provides a small **Click** CLI for quick smoke-tests::

    python data_module.py -c config.yaml -o data.batch_size=4
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Set
from copy import deepcopy

import click
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, load_dataset
from monai.transforms import (
    Compose,
    Lambda,
    Resize,
    ScaleIntensity,
    RandRotate,
    RandZoom,
    RandAdjustContrast,
    RandSpatialCrop,
    CenterSpatialCrop,
    RandFlip,
    RandGaussianNoise,
    RandGaussianSmooth,
    RandAffine,
    RandCoarseDropout,
    ToTensor,
    OneOf,
)
from torchvision.transforms import Normalize, ColorJitter
from monai.utils import set_determinism
from omegaconf import OmegaConf
from rich.console import Console

console = Console()

# --------------------------------------------------------------------------------------
#  Configuration schema (Ω-friendly)
# --------------------------------------------------------------------------------------

@dataclass
class Config:
    """Minimal subset needed for the datamodule (rest lives elsewhere)."""

    # —— I/O ——
    root: Optional[Path] = None
    tsv: Optional[Path] = None
    remote_id: Optional[str] = None

    # —— loader ——
    batch_size: int = 8
    num_workers: int = 4
    seed: int | None = 42

    # —— preprocessing ——
    interpolation: str = "bicubic"
    resize_size: int = 256
    crop_size: int = 224

    # —— augmentation strength ——
    rotate_deg: float = 15.0
    zoom_range: Tuple[float, float] = (0.9, 1.1)
    contrast_gamma: Tuple[float, float] = (0.9, 1.1)
    aug_prob: float = 0.5
    
    # —— autoencoder specific ——
    noise_level: float = 0.3
    mask_ratio: float = 0.4

    def mode(self) -> str:
        return "remote" if self.remote_id else "local"

# --------------------------------------------------------------------------------------
#  Utility helpers
# --------------------------------------------------------------------------------------

def set_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_determinism(seed=seed)

def hwc_to_chw(img: np.ndarray) -> np.ndarray:
    """Convert H×W×C → C×H×W for MONAI tensors."""
    return np.moveaxis(img, -1, 0).copy()

class DualTransform:
    """Wrapper to create two differently augmented versions of the same image.
    
    Used for autoencoder training where we need both input and target images
    with different augmentations.
    """
    def __init__(self, base_transform):
        self.base_transform = base_transform
    
    def __call__(self, img: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        # Create two separate transforms by deep copying
        transform1 = deepcopy(self.base_transform)
        transform2 = deepcopy(self.base_transform)
        
        # Apply different random augmentations to each
        return transform1(img), transform2(img)

def add_noise(img: torch.Tensor, noise_level: float) -> torch.Tensor:
    """Add Gaussian noise to the input image.
    
    Args:
        img: Input tensor image
        noise_level: Standard deviation of the noise
        
    Returns:
        Noisy image tensor, clamped to [0, 1]
    """
    if noise_level <= 0:
        return img
    
    noise = torch.randn_like(img) * noise_level
    return torch.clamp(img + noise, 0, 1)

def random_mask(img: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly mask image regions for MAE-style training.
    
    Args:
        img: Input tensor image [C, H, W]
        mask_ratio: Ratio of pixels to mask (0.0-1.0)
        
    Returns:
        Tuple of (masked image, binary mask)
    """
    if mask_ratio <= 0:
        return img, torch.ones_like(img)
    
    # Create random mask (1 = keep, 0 = mask)
    mask = torch.rand(img.shape[1:]) > mask_ratio  # [H, W]
    mask = mask.float()
    
    # Expand mask to match image channels
    mask = mask.expand_as(img)
    
    # Apply mask (multiply by mask to zero out masked regions)
    masked_img = img * mask
    
    return masked_img, mask

# --------------------------------------------------------------------------------------
#  Dataset builders
# --------------------------------------------------------------------------------------

def _build_local(cfg: Config) -> DatasetDict:
    data_path = cfg.root / cfg.tsv
    df = pd.read_csv(data_path, sep="\t")
    if "image" not in df.columns:
        df = df.rename(columns={"image_path": "image"})
    df["image"] = df["image"].apply(lambda p: str(cfg.root / p))

    feats = Features(
        {"image": Image(), "label": ClassLabel(names=sorted(df["label"].unique()))}
    )
    return DatasetDict(
        {
            split: Dataset.from_pandas(
                df[df["split"] == split][["image", "label"]],
                features=feats,
                preserve_index=False,
            )
            for split in ("train", "test")
        }
    )

def _build_remote(cfg: Config) -> DatasetDict:
    print("Loading remote dataset from:", cfg.remote_id)
    return load_dataset(cfg.remote_id, split={"train": "train", "test": "test"})

# --------------------------------------------------------------------------------------
#  MONAI transform factory
# --------------------------------------------------------------------------------------

def _make_transform(cfg: Config, train: bool):
    """Create transformation pipeline with dual augmentation for training.
    
    For training, this returns a DualTransform that produces two differently
    augmented versions of the same image. For validation/testing, it returns
    a standard deterministic transform.
    
    Args:
        cfg: Configuration object
        train: Whether to create training transforms (with augmentation)
        
    Returns:
        A transform or DualTransform
    """
    base = [
        Lambda(hwc_to_chw),
        Resize(
            (cfg.resize_size, cfg.resize_size),
            mode=cfg.interpolation,
            align_corners=None,
        ),
    ]
    
    if train:
        # Light augmentations always applied
        light = [
            RandRotate(
                range_x=np.deg2rad(cfg.rotate_deg), prob=cfg.aug_prob, keep_size=True
            ),
            RandZoom(
                min_zoom=cfg.zoom_range[0],
                max_zoom=cfg.zoom_range[1],
                prob=cfg.aug_prob,
                keep_size=True,
                mode=cfg.interpolation,
            ),
            RandAdjustContrast(prob=cfg.aug_prob, gamma=cfg.contrast_gamma),
            RandSpatialCrop((cfg.crop_size, cfg.crop_size), random_size=False),
            RandFlip(spatial_axis=0, prob=0.5),  # Horizontal flip
            RandFlip(spatial_axis=1, prob=0.5),  # Vertical flip
        ]
        
        # Heavy augmentations with random application
        heavy = Compose(
            [
                RandGaussianSmooth(
                    prob=cfg.aug_prob, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5)
                ),
                RandAffine(
                    prob=cfg.aug_prob,
                    rotate_range=np.deg2rad(15),
                    translate_range=(0.1, 0.1),
                    scale_range=(0.9, 1.1),
                    padding_mode="border",
                ),
                RandCoarseDropout(prob=cfg.aug_prob, holes=8, spatial_size=(16, 16)),
                # Additional color jitter for autoencoder robustness
                Lambda(lambda x: torch.from_numpy(x.astype(np.float32)).permute(1, 2, 0).numpy() 
                      if x.shape[0] <= 3 else x),  # Convert to HWC if needed
                Lambda(lambda x: ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.05
                )(torch.from_numpy(x.astype(np.float32)))),
                Lambda(lambda x: x.permute(2, 0, 1).numpy() 
                      if isinstance(x, torch.Tensor) and x.dim() == 3 else x),  # Convert back to CHW
            ]
        )
        
        # With 80% chance apply heavy; 20% return identity (i.e. skip them)
        skip_or_heavy = OneOf([heavy, Compose([])], weights=[0.8, 0.2])

        aug = light + [skip_or_heavy]
    else:
        aug = [CenterSpatialCrop((cfg.crop_size, cfg.crop_size))]

    # Final normalization pipeline
    normalize = [
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    transform = Compose(base + aug + normalize)
    
    # For training, wrap in DualTransform to get two augmented views
    return DualTransform(transform) if train else transform

# --------------------------------------------------------------------------------------
#  Lightning DataModule
# --------------------------------------------------------------------------------------

class RetinaDataModule(pl.LightningDataModule):
    """Wraps local TSV or HuggingFace dataset into a LightningDataModule."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.train_tf = _make_transform(cfg, train=True)
        self.test_tf = _make_transform(cfg, train=False)

    def prepare_data(self):
        if self.cfg.mode() == "remote":
            _build_remote(self.cfg)

    def setup(self, stage: Optional[str] = None):
        ds = (
            _build_remote(self.cfg)
            if self.cfg.mode() == "remote"
            else _build_local(self.cfg)
        )
        self.num_classes = len(ds["train"].features["label"].names)

        # Set transforms
        def _apply(batch, tf):
            imgs = batch["image"]
            if not isinstance(imgs, list):
                imgs = [imgs]
                single = True
            else:
                single = False
            
            # Apply transform to each image
            if isinstance(tf, DualTransform):
                # For training with dual transforms, we get tuple of (input, target)
                transformed = [tf(np.array(im)) for im in imgs]
                if single:
                    batch["pixel_values"] = transformed[0]  # Single (input, target) pair
                else:
                    # Unzip the list of tuples into two lists
                    input_views, target_views = zip(*transformed)
                    batch["pixel_values"] = (list(input_views), list(target_views))
            else:
                # For validation/testing, we just get a single tensor
                pix = [tf(np.array(im)) for im in imgs]
                batch["pixel_values"] = pix[0] if single else pix
            
            return batch

        ds["train"].set_transform(lambda b: _apply(b, self.train_tf))
        ds["test"].set_transform(lambda b: _apply(b, self.test_tf))
        self.ds_train, self.ds_test = ds["train"], ds["test"]

    def _collate(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function handling dual transforms and creating input/target pairs.
        
        This updated collate function handles the dual-transformed images for autoencoder
        training, creating separate tensors for input and target images.
        
        Args:
            batch: List of samples from the dataset
            
        Returns:
            Dictionary with 'input_images', 'target_images', and 'labels'
        """
        input_images = []
        target_images = []
        labels = []
        
        for b in batch:
            # Extract pixel values
            pixel_values = b["pixel_values"]
            
            # Handle both training and validation data formats
            if isinstance(pixel_values, tuple) and len(pixel_values) == 2:
                # We have input and target from dual transform
                inp_img, tgt_img = pixel_values
                
                # Add noise and masking to input image only (for autoencoder training)
                inp_img = add_noise(inp_img, self.cfg.noise_level)
                inp_img, _ = random_mask(inp_img, self.cfg.mask_ratio)
            else:
                # For validation/test where we don't have dual transforms
                # Use the same image for both input and target
                inp_img = tgt_img = pixel_values
            
            input_images.append(inp_img)
            target_images.append(tgt_img)
            labels.append(b["label"])
        
        return {
            "input_images": torch.stack(input_images),
            "target_images": torch.stack(target_images),
            "labels": torch.tensor(labels),
        }

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self._collate,
            worker_init_fn=lambda wid: (
                np.random.seed(self.cfg.seed + wid)
                if self.cfg.seed is not None
                else None
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            persistent_workers=True,
            collate_fn=self._collate,
            worker_init_fn=lambda wid: (
                np.random.seed(self.cfg.seed + wid)
                if self.cfg.seed is not None
                else None
            ),
        )

# --------------------------------------------------------------------------------------
#  CLI smoke-test
# --------------------------------------------------------------------------------------

@click.command(
    help="Quickly sanity-check the datamodule → shows batch shape and label counts."
)
@click.option(
    "--config", "-c", required=True, type=click.Path(exists=True, path_type=Path)
)
@click.option("--override", "-o", multiple=True, help="OmegaConf dot-list overrides")
@click.option(
    "--split",
    type=click.Choice(["train", "val"], case_sensitive=False),
    default="train",
)
def _main(config: Path, override: Tuple[str, ...], split: str):
    merged = OmegaConf.merge(
        OmegaConf.load(config), OmegaConf.from_dotlist(list(override))
    )
    cfg = Config(
        root=Path(merged.root).expanduser() if "root" in merged else None,
        tsv=Path(merged.tsv).expanduser() if "tsv" in merged else None,
        remote_id=merged.get("remote_id"),
        interpolation=str(merged.get("interpolation", "bicubic")),
        batch_size=int(merged.get("batch_size", 8)),
        num_workers=int(merged.get("num_workers", 4)),
        seed=None
        if str(merged.get("seed")) in {"-1", "None"}
        else int(merged.get("seed", 42)),
        resize_size=int(merged.get("resize_size", 256)),
        crop_size=int(merged.get("crop_size", 224)),
        noise_level=float(merged.get("noise_level", 0.3)),
        mask_ratio=float(merged.get("mask_ratio", 0.4)),
    )
    set_seed(cfg.seed)
    dm = RetinaDataModule(cfg)
    dm.prepare_data()
    dm.setup()
    dl = dm.train_dataloader() if split == "train" else dm.val_dataloader()
    batch = next(iter(dl))
    console.print(
        f"Loaded batch for [bold]{split}[/]:",
        "input_images:", batch["input_images"].shape,
        "target_images:", batch["target_images"].shape,
        "labels:", batch["labels"].shape,
    )

if __name__ == "__main__":
    _main()
