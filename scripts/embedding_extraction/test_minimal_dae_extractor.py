#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the MinimalDAEExtractor.

This script demonstrates the functionality of the MinimalDAEExtractor class, including:
1. Setting up sample data
2. Initializing and testing the extractor
3. Processing single batches and full datasets
4. Saving and loading embeddings
5. Verifying embedding quality
6. Handling errors and cleanup

Example usage:
    python test_minimal_dae_extractor.py
"""

import os
import sys
import tempfile
import logging
from pathlib import Path
import argparse
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Ensure the project root is in sys.path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the extractor
from src.models.embedding_extraction.minimal_dae.extractor import MinimalDAEExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_minimal_dae_extractor")


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing embedding extraction."""
    
    def __init__(self, num_samples: int = 50, image_size: int = 224):
        """
        Create a synthetic dataset.
        
        Args:
            num_samples: Number of samples to generate
            image_size: Size of the square images to generate
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.data = []
        self.labels = []
        
        # Generate synthetic data
        logger.info(f"Generating {num_samples} synthetic samples of size {image_size}x{image_size}")
        for i in range(num_samples):
            # Create a random RGB image (3 channels)
            img = torch.rand(3, image_size, image_size)
            
            # Create a label (0-4)
            label = torch.tensor(i % 5)
            
            self.data.append(img)
            self.labels.append(label)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


def create_mock_checkpoint(temp_dir: Path) -> Path:
    """
    Create a mock checkpoint file for testing.
    
    Args:
        temp_dir: Temporary directory to create the checkpoint in
        
    Returns:
        Path to the created checkpoint file
    """
    # Create a model with random weights
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 384, kernel_size=3, padding=1, stride=2),
        torch.nn.BatchNorm2d(384),
        torch.nn.ELU(),
        torch.nn.Conv2d(384, 128, kernel_size=3, padding=1, stride=2),
        torch.nn.BatchNorm2d(128),
        torch.nn.ELU(),
        torch.nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=2),
        torch.nn.BatchNorm2d(64),
        torch.nn.ELU()
    )
    
    # Create a state dict with the expected keys
    state_dict = {}
    for name, param in model.state_dict().items():
        # Rename keys to match expected format
        if "0." in name:
            new_name = name.replace("0.", "ae_encoder.encoder1.")
        elif "3." in name:
            new_name = name.replace("3.", "ae_encoder.encoder2.")
        elif "6." in name:
            new_name = name.replace("6.", "ae_encoder.encoder3.")
        else:
            new_name = name
        state_dict[new_name] = param
    
    # Add KAN layer
    state_dict["ae_encoder.kan.weight"] = torch.randn(64, 64, 5, 5)
    state_dict["ae_encoder.kan.bias"] = torch.randn(64)
    
    # Add ECA layer
    state_dict["ae_encoder.ECA_Net.0.weight"] = torch.randn(1, 1, 3)
    state_dict["ae_encoder.ECA_Net.0.bias"] = torch.randn(1)
    
    # Add decoder layers
    state_dict["ae_encoder.decoder1.0.weight"] = torch.randn(128)
    state_dict["ae_encoder.decoder1.0.bias"] = torch.randn(128)
    state_dict["ae_encoder.decoder1.1.weight"] = torch.randn(128, 64, 3, 3)
    state_dict["ae_encoder.decoder1.1.bias"] = torch.randn(128)
    state_dict["ae_encoder.decoder1.2.weight"] = torch.randn(128)
    state_dict["ae_encoder.decoder1.2.bias"] = torch.randn(128)
    
    state_dict["ae_encoder.decoder2.0.weight"] = torch.randn(384, 128, 3, 3)
    state_dict["ae_encoder.decoder2.0.bias"] = torch.randn(384)
    state_dict["ae_encoder.decoder2.1.weight"] = torch.randn(384)
    state_dict["ae_encoder.decoder2.1.bias"] = torch.randn(384)
    
    # Add bottleneck layers
    state_dict["bottleneck.encoder1.0.weight"] = torch.randn(384, 384, 3, 3)
    state_dict["bottleneck.encoder1.0.bias"] = torch.randn(384)
    state_dict["bottleneck.encoder1.1.weight"] = torch.randn(384)
    state_dict["bottleneck.encoder1.1.bias"] = torch.randn(384)
    
    state_dict["bottleneck.attn1.0.weight"] = torch.randn(384, 384, 1, 1)
    state_dict["bottleneck.attn1.0.bias"] = torch.randn(384)
    state_dict["bottleneck.attn1.1.weight"] = torch.randn(384)
    state_dict["bottleneck.attn1.1.bias"] = torch.randn(384)
    
    state_dict["bottleneck.encoder2.0.weight"] = torch.randn(16, 384, 3, 3)
    state_dict["bottleneck.encoder2.0.bias"] = torch.randn(16)
    state_dict["bottleneck.encoder2.1.weight"] = torch.randn(16)
    state_dict["bottleneck.encoder2.1.bias"] = torch.randn(16)
    
    state_dict["bottleneck.attn2.0.weight"] = torch.randn(16, 16, 1, 1)
    state_dict["bottleneck.attn2.0.bias"] = torch.randn(16)
    state_dict["bottleneck.attn2.1.weight"] = torch.randn(16)
    state_dict["bottleneck.attn2.1.bias"] = torch.randn(16)
    
    # Save the state dict to a file
    checkpoint_path = temp_dir / "mock_checkpoint.pth"
    torch.save({"state_dict": state_dict}, checkpoint_path)
    
    logger.info(f"Created mock checkpoint at {checkpoint_path}")
    return checkpoint_path


def test_single_batch(extractor: MinimalDAEExtractor, batch_size: int = 4) -> None:
    """
    Test processing a single batch.
    
    Args:
        extractor: The initialized extractor
        batch_size: Batch size to use
    """
    logger.info(f"\n{'='*50}\nTesting single batch processing\n{'='*50}")
    
    # Create a sample batch
    batch = torch.rand(batch_size, 3, 224, 224)
    
    try:
        # Process batch and time it
        logger.info(f"Processing batch of shape {batch.shape}")
        embeddings = extractor.process_batch(batch)
        
        # Check embeddings shape
        expected_dim = 4096  # 16 x 16 x 16
        if embeddings.shape == (batch_size, expected_dim):
            logger.info(f"✓ Embeddings shape is correct: {embeddings.shape}")
        else:
            logger.error(f"✗ Embeddings shape is incorrect: {embeddings.shape}, expected {(batch_size, expected_dim)}")
        
        # Check embeddings values
        logger.info(f"Embeddings stats: min={embeddings.min():.4f}, max={embeddings.max():.4f}, mean={embeddings.mean():.4f}")
        
        # Verify embeddings
        results = extractor.verify_embeddings(embeddings)
        logger.info(f"Verification results: {results}")
        
    except Exception as e:
        logger.error(f"Error in single batch test: {e}")
        raise


def test_dataset_processing(extractor: MinimalDAEExtractor, temp_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Test processing a complete dataset.
    
    Args:
        extractor: The initialized extractor
        temp_dir: Temporary directory for saving outputs
        
    Returns:
        Tuple of (embeddings, labels) from the dataset
    """
    logger.info(f"\n{'='*50}\nTesting dataset processing\n{'='*50}")
    
    # Create a synthetic dataset
    dataset = SyntheticDataset(num_samples=20, image_size=224)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Process the dataset
    save_path = temp_dir / "embeddings.npz"
    try:
        logger.info(f"Processing dataset with {len(dataset)} samples, saving to {save_path}")
        embeddings, labels = extractor.process_dataset(
            dataloader, 
            save_path=save_path,
            show_progress=True, 
            checkpoint_freq=2,
            metadata={"test_dataset": "synthetic"}
        )
        
        # Check results
        logger.info(f"Processed {len(embeddings)} samples with embedding dimension {embeddings.shape[1]}")
        
        return embeddings, labels
        
    except Exception as e:
        logger.error(f"Error in dataset processing test: {e}")
        raise


def test_save_load_embeddings(extractor: MinimalDAEExtractor, embeddings: np.ndarray, labels: np.ndarray, temp_dir: Path) -> None:
    """
    Test saving and loading embeddings.
    
    Args:
        extractor: The initialized extractor
        embeddings: Embeddings to save
        labels: Labels to save
        temp_dir: Temporary directory for saving outputs
    """
    logger.info(f"\n{'='*50}\nTesting embedding save/load\n{'='*50}")
    
    # Test saving in different formats
    formats = [".npz", ".csv"]
    
    for fmt in formats:
        save_path = temp_dir / f"embeddings{fmt}"
        
        try:
            # Save embeddings
            logger.info(f"Saving embeddings to {save_path}")
            metadata = {
                "test_key": "test_value",
                "dimensions": embeddings.shape[1]
            }
            extractor.save_embeddings(embeddings, labels, save_path, metadata=metadata)
            
            # Load embeddings
            logger.info(f"Loading embeddings from {save_path}")
            loaded_embeddings, loaded_labels, loaded_metadata = extractor.load_embeddings(save_path)
            
            # Verify loaded data
            assert np.array_equal(embeddings, loaded_embeddings), "Loaded embeddings don't match originals"
            assert np.array_equal(labels, loaded_labels), "Loaded labels don't match originals"
            assert "test_key" in loaded_metadata, "Metadata not loaded correctly"
            
            logger.info(f"✓ Successfully saved and loaded embeddings in {fmt} format")
            logger.info(f"Loaded metadata: {loaded_metadata}")
            
        except Exception as e:
            logger.error(f"Error in save/load test for {fmt}: {e}")
            raise


def test_embedding_verification(extractor: MinimalDAEExtractor, embeddings: np.ndarray) -> None:
    """
    Test embedding verification functionality.
    
    Args:
        extractor: The initialized extractor
        embeddings: Embeddings to verify
    """
    logger.info(f"\n{'='*50}\nTesting embedding verification\n{'='*50}")
    
    try:
        # Basic verification
        results = extractor.verify_embeddings(embeddings, check_clustering=True)
        logger.info(f"Verification results: {results}")
        
        # Test with corrupted embeddings
        corrupt_embeddings = embeddings.copy()
        
        # Add some NaN values
        corrupt_embeddings[0, 0] = np.nan
        
        # Add some extreme values
        corrupt_embeddings[1, 1] = 1000.0
        
        # Verify corrupted embeddings
        corrupt_results = extractor.verify_embeddings(corrupt_embeddings)
        logger.info(f"Corrupt embeddings verification results: {corrupt_results}")
        
        # Should detect issues
        assert not corrupt_results['valid'], "Verification should detect issues in corrupt embeddings"
        assert len(corrupt_results['issues']) > 0, "Verification should report issues"
        
        logger.info(f"✓ Embedding verification correctly identified issues")
        
    except Exception as e:
        logger.error(f"Error in verification test: {e}")
        raise


def test_error_handling_and_cleanup(extractor: MinimalDAEExtractor) -> None:
    """
    Test error handling and resource cleanup.
    
    Args:
        extractor: The initialized extractor
    """
    logger.info(f"\n{'='*50}\nTesting error handling and cleanup\n{'='*50}")
    
    try:
        # Test with invalid input shapes
        invalid_batch = torch.rand(4, 1, 224, 224)  # Wrong number of channels
        
        try:
            embeddings = extractor.process_batch(invalid_batch)
            logger.error("✗ Failed to catch invalid input shape")
        except ValueError as e:
            logger.info(f"✓ Correctly caught invalid input: {e}")
        
        # Test context manager for cleanup
        logger.info("Testing context manager for cleanup")
        with MinimalDAEExtractor(extractor.model_path, extractor.device) as temp_extractor:
            temp_extractor.load_model()
            # Process a small batch
            batch = torch.rand(2, 3, 224, 224)
            embeddings = temp_extractor.process_batch(batch)
            logger.info(f"Processed batch inside context manager, shape: {embeddings.shape}")
        
        # Model should be cleaned up after context exit
        assert temp_extractor.model is None, "Model should be None after context exit"
        logger.info("✓ Context manager successfully cleaned up resources")
        
    except Exception as e:
        logger.error(f"Error in error handling test: {e}")
        raise


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test the MinimalDAEExtractor functionality")
    parser.add_argument("--checkpoint", type=str, help="Path to existing model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to run tests on (cuda/cpu)")
    args = parser.parse_args()
    
    logger.info(f"Running tests on device: {args.device}")
    
    # Create a temporary directory for test outputs
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        logger.info(f"Using temporary directory: {temp_dir}")
        
        # Get or create checkpoint
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
            logger.info(f"Using existing checkpoint: {checkpoint_path}")
        else:
            checkpoint_path = create_mock_checkpoint(temp_dir)
        
        # Initialize the extractor
        extractor = MinimalDAEExtractor(
            model_path=checkpoint_path,
            device=args.device,
            cache_dir=temp_dir
        )
        
        try:
            # Load the model
            extractor.load_model()
            
            # Test single batch processing
            test_single_batch(extractor)
            
            # Test dataset processing
            embeddings, labels = test_dataset_processing(extractor, temp_dir)
            
            # Test saving and loading embeddings
            test_save_load_embeddings(extractor, embeddings, labels, temp_dir)
            
            # Test embedding verification
            test_embedding_verification(extractor, embeddings)
            
            # Test error handling and cleanup
            test_error_handling_and_cleanup(extractor)
            
            logger.info(f"\n{'='*50}\nAll tests completed successfully!\n{'='*50}")
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
            raise
        finally:
            # Clean up
            extractor.cleanup()


if __name__ == "__main__":
    main()
