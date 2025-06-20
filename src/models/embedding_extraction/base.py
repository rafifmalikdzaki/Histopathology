#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base module for embedding extraction functionality.

This module provides abstract base classes and interfaces for embedding extractors
used in histopathology image analysis.
"""

import abc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaseEmbeddingExtractor(abc.ABC):
    """
    Abstract base class for all embedding extractors.
    
    This class defines the common interface that all embedding extractors
    must implement for consistent usage across the project.
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        """
        Initialize the embedding extractor.
        
        Args:
            model_path: Path to the model weights file.
            device: Device to run the model on ("cuda" or "cpu").
            **kwargs: Additional keyword arguments for specific extractor implementations.
        """
        self.model_path = Path(model_path) if model_path else None
        self.device = device
        self.model = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(f"Initializing {self.__class__.__name__} on {device}")
        
    @abc.abstractmethod
    def load_model(self) -> None:
        """
        Load the model weights from the specified path.
        
        This method must be implemented by all subclasses to load their
        specific model architecture and weights.
        
        Raises:
            FileNotFoundError: If the model file doesn't exist.
            RuntimeError: If there's an error loading the model.
        """
        pass
    
    @abc.abstractmethod
    def extract_embeddings(
        self, 
        inputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract embeddings from input data.
        
        Args:
            inputs: Input tensor to extract embeddings from.
            
        Returns:
            Tensor containing the extracted embeddings.
            
        Raises:
            RuntimeError: If there's an error during embedding extraction.
        """
        pass
    
    def process_batch(
        self, 
        batch: torch.Tensor
    ) -> np.ndarray:
        """
        Process a single batch of inputs and extract embeddings.
        
        Args:
            batch: A batch of input data.
            
        Returns:
            Numpy array of embeddings for the batch.
        """
        self.logger.debug(f"Processing batch of shape {batch.shape}")
        with torch.no_grad():
            batch = batch.to(self.device)
            embeddings = self.extract_embeddings(batch)
            return embeddings.cpu().numpy()
    
    def process_dataset(
        self,
        dataloader: DataLoader,
        save_path: Optional[Union[str, Path]] = None,
        show_progress: bool = True,
        checkpoint_freq: int = 50  # Save every 50 batches
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process an entire dataset and extract embeddings.
        
        Args:
            dataloader: PyTorch DataLoader to iterate over batches.
            save_path: Optional path to save embeddings.
            show_progress: Whether to show progress bar.
            checkpoint_freq: Frequency (in batches) to save intermediate results.
            
        Returns:
            Tuple of (embeddings, labels) arrays.
            
        Raises:
            RuntimeError: If there's an error during processing.
        """
        if self.model is None:
            self.logger.info("Model not loaded, loading now...")
            self.load_model()
            
        self.logger.info(f"Processing dataset with {len(dataloader)} batches")
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(dataloader, desc="Extracting embeddings")
            except ImportError:
                self.logger.warning("tqdm not installed, disabling progress bar")
                iterator = dataloader
                show_progress = False
        else:
            iterator = dataloader
        
        # Implementation should be provided by subclasses for specific
        # checkpointing and interrupt handling behavior
        raise NotImplementedError(
            "process_dataset must be implemented by subclasses"
        )
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        save_path: Union[str, Path]
    ) -> None:
        """
        Save embeddings and labels to a file.
        
        Args:
            embeddings: Numpy array of embeddings.
            labels: Numpy array of corresponding labels.
            save_path: Path to save the embeddings to.
            
        Raises:
            IOError: If there's an error saving the embeddings.
        """
        save_path = Path(save_path)
        self.logger.info(f"Saving embeddings to {save_path}")
        
        try:
            if save_path.suffix == '.npz':
                np.savez(save_path, embeddings=embeddings, labels=labels)
            elif save_path.suffix == '.csv':
                import pandas as pd
                df = pd.DataFrame(embeddings)
                df['label'] = labels
                df.to_csv(save_path, index=False)
            else:
                self.logger.warning(
                    f"Unknown file extension: {save_path.suffix}, saving as .npz"
                )
                np.savez(save_path, embeddings=embeddings, labels=labels)
        except Exception as e:
            self.logger.error(f"Error saving embeddings: {e}")
            raise IOError(f"Failed to save embeddings: {e}")
