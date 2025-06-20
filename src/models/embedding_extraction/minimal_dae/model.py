#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Minimal DAE model implementation for embedding extraction.

This module provides a simplified version of the DAE model that includes only
the components needed for embedding extraction (encoder and bottleneck).
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Configure logging
logger = logging.getLogger(__name__)

class MinimalBottleneck(nn.Module):
    """
    Minimal implementation of the bottleneck module.
    
    This class implements only the essential components needed for embedding extraction,
    matching the architecture in the original checkpoint.
    """
    def __init__(self):
        super(MinimalBottleneck, self).__init__()

        # First encoder in bottleneck: 384 -> 384
        self.encoder1 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(384),
            nn.ELU(inplace=True),
        )

        # Attention for first encoder (simplified)
        self.attn1 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=1, padding=0),
            nn.BatchNorm2d(384),
            nn.Sigmoid()
        )

        # Second encoder in bottleneck: 384 -> 16
        self.encoder2 = nn.Sequential(
            nn.Conv2d(384, 16, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
        )

        # Attention for second encoder (simplified)
        self.attn2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=1, padding=0),
            nn.BatchNorm2d(16),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the bottleneck to get embeddings.
        
        Args:
            x: Input tensor (batch_size, 384, H/4, W/4)
            
        Returns:
            z: Bottleneck encoding (batch_size, 16, H/16, W/16)
        """
        x = self.encoder1(x)
        x = x * self.attn1(x)  # Simple attention mechanism
        x = self.encoder2(x)
        z = x * self.attn2(x)  # Simple attention mechanism
        
        return z


class MinimalDAEModel(nn.Module):
    """
    Minimal implementation of the DAE model focused on embedding extraction.
    
    This class implements only the encoder and bottleneck components needed
    for generating embeddings, matching the architecture in the original checkpoint.
    """
    
    def __init__(self, device: str = 'cuda'):
        super(MinimalDAEModel, self).__init__()
        
        # Initial encoder layers: 3 -> 384 -> 128 -> 64
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 384, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(384),
            nn.ELU(inplace=True),
        )
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
        )
        
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True)
        )
        
        # Simplified KAN layer (as a regular conv with identity function)
        self.kan = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        
        # Simplified ECA attention
        self.eca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv1d(1, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Decoder part of the encoder: 64 -> 128 -> 384
        self.decoder1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 128, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True), 
        )
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 384, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(384),
            nn.ELU(inplace=True), 
        )
        
        # Bottleneck module
        self.bottleneck = MinimalBottleneck()
        
        self.device = device
        self.to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to generate embeddings.
        
        Args:
            x: Input tensor (batch_size, 3, H, W)
            
        Returns:
            z: Embedding tensor from bottleneck
        """
        # Encode
        x = self.encoder1(x)
        residual1 = x  # Store first residual
        
        x = self.encoder2(x)
        residual2 = x  # Store second residual
        
        x = self.encoder3(x)
        
        # Apply KAN and attention
        x = self.kan(x)
        
        # Apply ECA attention
        b, c, h, w = x.shape
        # Step 1: Apply adaptive avg pooling (first layer of eca)
        y = self.eca[0](x)  # Shape: [b, c, 1, 1]
        # Step 2: Reshape for Conv1d - squeeze spatial dims and transpose
        y = y.squeeze(-1).squeeze(-1).unsqueeze(1)  # Shape: [b, 1, c]
        # Step 3: Apply Conv1d and Sigmoid (remaining layers of eca)
        y = self.eca[1:](y)  # Shape: [b, 1, c]
        # Step 4: Reshape to broadcast with x - expand spatial dims
        y = y.squeeze(1).view(b, c, 1, 1)  # Shape: [b, c, 1, 1]
        # Step 5: Apply attention - element-wise multiplication
        x = x * y
        
        # Decoder part of encoder
        x = self.decoder1(x) + residual2
        x = self.decoder2(x) + residual1
        
        # Get embedding from bottleneck
        z = self.bottleneck(x)
        
        return z
    
    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from input tensor.
        
        Args:
            x: Input tensor (batch_size, 3, H, W)
            
        Returns:
            Flattened embedding tensor with shape (batch_size, 4096)
        """
        z = self.forward(x)
        # Apply adaptive pooling to get fixed spatial dimensions (16x16)
        z = F.adaptive_avg_pool2d(z, (16, 16))
        # Flatten the spatial dimensions to get 4096-dimensional embeddings (16*16*16)
        return z.reshape(z.size(0), -1)


def load_minimal_model(
    checkpoint_path: Union[str, Path], 
    device: str = 'cuda'
) -> MinimalDAEModel:
    """
    Load a minimal DAE model from a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Device to load the model on.
        
    Returns:
        Loaded minimal DAE model.
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
        RuntimeError: If loading fails.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    logger.info(f"Loading minimal DAE model from {checkpoint_path}")
    
    try:
        # Create model
        model = MinimalDAEModel(device=device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Process state dict to match our simplified model
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Clean up state dict keys (remove "model." prefix if present)
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("model.", "")
            # Only include keys that match our minimal model
            if any(new_key.startswith(prefix) for prefix in [
                "ae_encoder.encoder", 
                "ae_encoder.kan",
                "ae_encoder.ECA_Net",
                "ae_encoder.decoder",
                "bottleneck.encoder",
                "bottleneck.attn"
            ]):
                # Map keys to our simplified model
                new_key = new_key.replace("ae_encoder.", "")
                new_key = new_key.replace("bottleneck.", "bottleneck.")
                new_key = new_key.replace("ECA_Net", "eca")
                # Try to load the parameter if it matches
                try:
                    param_size = model.state_dict()[new_key].size()
                    if param_size == value.size():
                        new_state_dict[new_key] = value
                    else:
                        logger.warning(f"Skipping parameter {new_key}: size mismatch")
                except KeyError:
                    # Skip parameters that don't exist in our model
                    logger.debug(f"Skipping parameter {new_key}: not in model")
        
        # Load the filtered state dict
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing keys in state dict: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in state dict: {unexpected_keys}")
        
        model.eval()  # Set to evaluation mode
        return model
    
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Failed to load model from {checkpoint_path}: {e}")
