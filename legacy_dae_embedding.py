import os
import sys
import torch
import torch.utils.data
import pandas as pd
import numpy as np
import logging
import shutil
from pathlib import Path
from typing import Union, Optional, List, Tuple, Dict, Any
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

# Import the legacy model
from legacy_dae_kan_model import LegacyDAE_KAN_Attention

# Set up proper imports with error handling
try:
    from histopathology.models.autoencoders.dae_kan_attention.histopathology_dataset import create_dataset, ImageDataset
    DATASET_AVAILABLE = True
except ImportError:
    DATASET_AVAILABLE = False
    logging.warning("Could not import dataset module. Make sure PYTHONPATH includes the project root.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("legacy_dae_embedding.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LegacyDAEEmbeddingExtractor:
    """
    A class to extract embeddings from the legacy DAE-KAN-Attention model.
    
    This class is specifically designed to be compatible with the checkpoint
    architecture, handling the differences in model structure.
    
    Args:
        checkpoint_path (str or Path): Path to the checkpoint file
        device (str, optional): Device to run the model on. Defaults to 'cuda' if available
        batch_size (int, optional): Batch size for processing. Defaults to 32
    """
    
    def __init__(
        self, 
        checkpoint_path: Union[str, Path], 
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Hook for bottleneck features
        self.bottleneck_features = None
        self._register_hooks()
        
        logger.info(f"Initialized LegacyDAEEmbeddingExtractor with device={self.device}, batch_size={self.batch_size}")
    
    def _load_model(self) -> LegacyDAE_KAN_Attention:
        """
        Load the legacy DAE-KAN-Attention model from checkpoint with non-strict matching.
        
        Returns:
            LegacyDAE_KAN_Attention: The loaded model
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If model loading fails
        """
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")
        
        try:
            logger.info(f"Loading model from checkpoint: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Create model
            model = LegacyDAE_KAN_Attention(device=self.device)
            
            # Adapt to load weights in a non-strict manner
            new_state_dict = {}
            model_dict = model.state_dict()
            mismatch_count = 0
            match_count = 0
            
            # Process checkpoint state dict
            for key, value in checkpoint['state_dict'].items():
                new_key = key.replace("model.", "")
                
                # Check if key exists and shapes match
                if new_key in model_dict:
                    if value.shape == model_dict[new_key].shape:
                        new_state_dict[new_key] = value
                        match_count += 1
                    else:
                        mismatch_count += 1
                        logger.warning(f"Shape mismatch for {new_key}: checkpoint={value.shape}, model={model_dict[new_key].shape}")
            
            # Load filtered state dict with strict=False to allow missing keys
            model.load_state_dict(new_state_dict, strict=False)
            model.to(self.device)
            
            logger.info(f"Model loaded with {match_count} matching and {mismatch_count} mismatched parameters")
            logger.info("Note: Mismatched layers are initialized with random weights")
            return model
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def _register_hooks(self):
        """
        Register forward hook to capture bottleneck features.
        """
        def bottleneck_hook(module, input, output):
            # Capture the bottleneck output (z)
            self.bottleneck_features = output[1]
        
        # Register hook on the bottleneck module to capture 'z'
        self.model.bottleneck.register_forward_hook(bottleneck_hook)
        logger.info("Registered hook on bottleneck layer")
    
    def extract_embeddings(
        self, 
        data: torch.Tensor,
        flatten: bool = True
    ) -> torch.Tensor:
        """
        Extract embeddings from input data.
        
        Args:
            data (torch.Tensor): Input tensor of shape (B, C, H, W)
            flatten (bool, optional): Whether to flatten spatial dimensions. Defaults to True.
            
        Returns:
            torch.Tensor: Embeddings tensor
            
        Raises:
            ValueError: If input shape is invalid
            RuntimeError: If embedding extraction fails
        """
        # Validate input
        if not isinstance(data, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
        
        if len(data.shape) != 4:
            raise ValueError(f"Expected 4D input tensor (B,C,H,W), got shape {data.shape}")
        
        try:
            # Move data to device
            data = data.to(self.device)
            
            # Forward pass with no gradient calculation
            with torch.no_grad():
                _, _, _ = self.model(data)
                
                # Get bottleneck features captured by the hook
                if self.bottleneck_features is None:
                    raise RuntimeError("Failed to capture bottleneck features")
                
                embeddings = self.bottleneck_features
                
                # Flatten if requested
                if flatten:
                    embeddings = embeddings.view(embeddings.size(0), -1)
                
                return embeddings.cpu()
                
        except Exception as e:
            if "CUDA out of memory" in str(e):
                logger.error(f"GPU out of memory. Try reducing batch size (current: {self.batch_size})")
                raise RuntimeError(f"GPU out of memory. Try reducing batch size (current: {self.batch_size})")
            
            logger.error(f"Error extracting embeddings: {str(e)}")
            raise RuntimeError(f"Failed to extract embeddings: {str(e)}")
    
    def process_dataset(
        self,
        dataloader: torch.utils.data.DataLoader,
        save_path: Optional[Union[str, Path]] = None,
        show_progress: bool = True,
        checkpoint_freq: int = 50  # Save checkpoint every 50 batches
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process an entire dataset and extract embeddings with checkpointing and interrupt handling.
        
        Args:
            dataloader (DataLoader): PyTorch dataloader
            save_path (str or Path, optional): Path to save embeddings CSV. Defaults to None.
            show_progress (bool, optional): Whether to show progress bar. Defaults to True.
            checkpoint_freq (int, optional): How often to save checkpoints (in batches). Defaults to 50.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of (embeddings, labels)
        """
        embeddings = []
        labels = []
        
        # Create temporary directory for checkpoints
        temp_dir = Path("temp_embeddings")
        temp_dir.mkdir(exist_ok=True)
        logger.info(f"Created temporary directory for checkpoints: {temp_dir}")
        
        iterator = dataloader
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting embeddings")
        
        try:
            with torch.no_grad():
                for batch_idx, (data, label) in enumerate(iterator):
                    # Move data to device
                    data = data.to(self.device)
                    
                    # Extract embeddings
                    batch_embeddings = self.extract_embeddings(data)
                    
                    # Store results
                    embeddings.append(batch_embeddings.numpy())
                    labels.append(label.cpu().numpy())
                    
                    # Save checkpoint periodically
                    if batch_idx > 0 and batch_idx % checkpoint_freq == 0:
                        self._save_checkpoint(
                            embeddings, 
                            labels, 
                            temp_dir / f"checkpoint_{batch_idx}.npz",
                            batch_idx
                        )
                    
                    # Free up GPU memory
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
            
            # Concatenate results
            embeddings = np.concatenate(embeddings, axis=0)
            labels = np.concatenate(labels, axis=0)
            
            logger.info(f"Extracted embeddings with shape: {embeddings.shape}")
            
            # Save to CSV if requested
            if save_path:
                save_path = Path(save_path)
                df = pd.DataFrame(embeddings)
                df['label'] = labels
                df.to_csv(save_path, index=False)
                logger.info(f"Saved embeddings to {save_path}")
            
            # Clean up temporary files
            self._cleanup_temp_dir(temp_dir)
            
            return embeddings, labels
            
        except KeyboardInterrupt:
            logger.warning("Process interrupted by user. Saving partial results...")
            try:
                # Save partial results
                if embeddings and labels:
                    partial_embeddings = np.concatenate(embeddings, axis=0)
                    partial_labels = np.concatenate(labels, axis=0)
                    
                    if save_path:
                        save_path = Path(save_path)
                        partial_path = save_path.with_stem(f"{save_path.stem}_partial")
                        df = pd.DataFrame(partial_embeddings)
                        df['label'] = partial_labels
                        df.to_csv(partial_path, index=False)
                        logger.info(f"Saved partial results to {partial_path}")
            except Exception as e:
                logger.error(f"Error saving partial results: {e}")
            
            # Don't delete temp files in case they're needed for recovery
            logger.info(f"Temporary checkpoint files preserved in {temp_dir}")
            raise
            
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            raise RuntimeError(f"Failed to process dataset: {str(e)}")
        
    def _save_checkpoint(self, embeddings_list, labels_list, checkpoint_path, batch_idx):
        """Save a checkpoint of the current embeddings and labels."""
        try:
            # Concatenate current results
            temp_embeddings = np.concatenate(embeddings_list, axis=0)
            temp_labels = np.concatenate(labels_list, axis=0)
            
            # Save to numpy compressed format
            np.savez(
                checkpoint_path,
                embeddings=temp_embeddings,
                labels=temp_labels
            )
            logger.info(f"Saved checkpoint at batch {batch_idx} to {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def _cleanup_temp_dir(self, temp_dir):
        """Clean up temporary checkpoint directory."""
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up temporary directory: {e}")
    
    def process_in_batches(
        self,
        data: torch.Tensor,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Process large input tensor in batches.
        
        Args:
            data (torch.Tensor): Input tensor of shape (N, C, H, W)
            show_progress (bool, optional): Whether to show progress bar. Defaults to True.
            
        Returns:
            np.ndarray: Embeddings array
        """
        if len(data) == 0:
            return np.array([])
        
        embeddings_list = []
        num_samples = len(data)
        
        iterator = range(0, num_samples, self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Processing batches")
        
        try:
            for i in iterator:
                # Get batch
                batch = data[i:min(i + self.batch_size, num_samples)]
                
                # Extract embeddings
                batch_embeddings = self.extract_embeddings(batch)
                embeddings_list.append(batch_embeddings.numpy())
                
                # Free up GPU memory
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
            
            # Concatenate results
            embeddings = np.concatenate(embeddings_list, axis=0)
            logger.info(f"Processed {num_samples} samples in batches, final shape: {embeddings.shape}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error processing in batches: {str(e)}")
            raise RuntimeError(f"Failed to process in batches: {str(e)}")


def create_dataloader(
    dataset_name: str, 
    batch_size: int = 64,
    num_workers: int = 4
) -> torch.utils.data.DataLoader:
    """
    Create a dataloader for a specific dataset.
    
    Args:
        dataset_name (str): Name of the dataset to load
        batch_size (int, optional): Batch size. Defaults to 64.
        num_workers (int, optional): Number of workers. Defaults to 4.
        
    Returns:
        DataLoader: PyTorch dataloader
    """
    try:
        # Import dataset functions here to handle potential import errors
        from histopathology.models.autoencoders.dae_kan_attention.histopathology_dataset import create_dataset, ImageDataset
        
        # Create dataset
        data = create_dataset(dataset_name)
        dataset = ImageDataset(*data, device=None)  # Don't move to device, let DataLoader handle it
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True  # Faster data transfer to GPU
        )
        
        logger.info(f"Created dataloader for dataset '{dataset_name}' with {len(dataset)} samples")
        return dataloader
        
    except Exception as e:
        logger.error(f"Error creating dataloader: {str(e)}")
        raise RuntimeError(f"Failed to create dataloader: {str(e)}")


def main():
    """Main function to extract embeddings."""
    parser = argparse.ArgumentParser(description="Extract embeddings from legacy DAE-KAN-Attention model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--output", type=str, default="embeddings.csv", help="Output CSV path")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--checkpoint-freq", type=int, default=50, help="How often to save checkpoints (in batches)")
    
    args = parser.parse_args()
    
    try:
        # Check if dataset module is available
        if not DATASET_AVAILABLE:
            logger.error("Dataset module not available!")
            logger.error("Please ensure PYTHONPATH includes the project root directory.")
            logger.error("Try running: PYTHONPATH=/home/dzakirm/Research/Histopathology python legacy_dae_embedding.py ...")
            return 1
            
        # Create dataset
        logger.info(f"Creating dataset: {args.dataset}")
        data = create_dataset(args.dataset)
        dataset = ImageDataset(*data, device=None)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # Initialize embedding extractor
        extractor = LegacyDAEEmbeddingExtractor(
            checkpoint_path=args.checkpoint,
            device=args.device,
            batch_size=args.batch_size
        )
        
        # Process dataset
        embeddings, labels = extractor.process_dataset(
            dataloader=dataloader,
            save_path=args.output,
            checkpoint_freq=args.checkpoint_freq
        )
        
        logger.info(f"Successfully extracted embeddings for {len(embeddings)} samples")
        logger.info(f"Embeddings shape: {embeddings.shape}")
        logger.info(f"Saved to: {args.output}")
        
    except ImportError as e:
        logger.error(f"Import error: {str(e)}")
        logger.error("Make sure PYTHONPATH includes the project root directory.")
        logger.error("Try running: PYTHONPATH=/home/dzakirm/Research/Histopathology python legacy_dae_embedding.py ...")
        return 1
    except KeyboardInterrupt:
        logger.error("Process interrupted by user.")
        logger.info("Partial results may have been saved. Check the logs for details.")
        return 130  # Standard exit code for SIGINT
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        return 1
    except RuntimeError as e:
        logger.error(f"Runtime error: {str(e)}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
