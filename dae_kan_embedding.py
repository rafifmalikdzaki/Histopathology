import os
import torch
import torch.utils.data
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Union, Optional, List, Tuple, Dict, Any
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Import the model and dataset
from histopathology.models.autoencoders.dae_kan_attention.model import DAE_KAN_Attention
try:
    from histopathology.models.autoencoders.dae_kan_attention.histopathology_dataset import create_dataset, ImageDataset
except ImportError:
    logging.warning("Could not import dataset classes. You'll need to provide your own dataset.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dae_embedding.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DAEEmbeddingExtractor:
    """
    A class to extract embeddings from the DAE-KAN-Attention model.
    
    This class loads a pre-trained DAE-KAN-Attention model and provides methods
    to extract embeddings from the bottleneck layer, which can be used for 
    clustering, visualization, or other downstream tasks.
    
    Args:
        checkpoint_path (str or Path): Path to the checkpoint file
        device (str, optional): Device to run the model on. Defaults to 'cuda' if available
        batch_size (int, optional): Batch size for processing. Defaults to 64
    """
    
    def __init__(
        self, 
        checkpoint_path: Union[str, Path], 
        device: Optional[str] = None,
        batch_size: int = 64
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
        
        logger.info(f"Initialized DAEEmbeddingExtractor with device={self.device}, batch_size={self.batch_size}")
    
    def _load_model(self) -> DAE_KAN_Attention:
        """
        Load the DAE-KAN-Attention model from checkpoint.
        
        Returns:
            DAE_KAN_Attention: The loaded model
            
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
            model = DAE_KAN_Attention(device=self.device)
            
            # Load weights, handling 'model.' prefix if present
            new_state_dict = {}
            for key, value in checkpoint['state_dict'].items():
                new_key = key.replace("model.", "")
                new_state_dict[new_key] = value
            
            model.load_state_dict(new_state_dict)
            model.to(self.device)
            logger.info("Model loaded successfully")
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
            self.bottleneck_features = output
        
        # Register hook on the bottleneck module to capture 'z'
        # This will be called during the forward pass
        self.model.bottleneck.register_forward_hook(
            lambda module, input, output: setattr(self, 'bottleneck_features', output[1])
        )
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
        show_progress: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process an entire dataset and extract embeddings.
        
        Args:
            dataloader (DataLoader): PyTorch dataloader
            save_path (str or Path, optional): Path to save embeddings CSV. Defaults to None.
            show_progress (bool, optional): Whether to show progress bar. Defaults to True.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of (embeddings, labels)
        """
        embeddings = []
        labels = []
        
        iterator = dataloader
        if show_progress:
            iterator = tqdm(dataloader, desc="Extracting embeddings")
        
        try:
            with torch.no_grad():
                for data, label in iterator:
                    # Move data to device
                    data = data.to(self.device)
                    
                    # Extract embeddings
                    batch_embeddings = self.extract_embeddings(data)
                    
                    # Store results
                    embeddings.append(batch_embeddings.numpy())
                    labels.append(label.cpu().numpy())
                    
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
            
            return embeddings, labels
            
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            raise RuntimeError(f"Failed to process dataset: {str(e)}")
    
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
    
    def visualize_embeddings(
        self,
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        method: str = 'tsne',
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """
        Visualize embeddings using dimensionality reduction.
        
        Args:
            embeddings (np.ndarray): Embeddings array
            labels (np.ndarray, optional): Labels for coloring points. Defaults to None.
            method (str, optional): Reduction method ('tsne', 'pca'). Defaults to 'tsne'.
            figsize (tuple, optional): Figure size. Defaults to (10, 8).
            save_path (str or Path, optional): Path to save figure. Defaults to None.
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        if len(embeddings) == 0:
            logger.warning("Empty embeddings array, cannot visualize")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No data to visualize", ha='center', va='center')
            return fig
        
        # Reduce dimensions
        if method.lower() == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
            reduced_data = reducer.fit_transform(embeddings)
        elif method.lower() == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            reduced_data = reducer.fit_transform(embeddings)
        else:
            logger.warning(f"Unknown method: {method}, using TSNE")
            reducer = TSNE(n_components=2, random_state=42)
            reduced_data = reducer.fit_transform(embeddings)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot with or without labels
        if labels is not None:
            scatter = ax.scatter(
                reduced_data[:, 0], 
                reduced_data[:, 1], 
                c=labels, 
                cmap='tab10', 
                alpha=0.7
            )
            legend = ax.legend(*scatter.legend_elements(), title="Classes")
            ax.add_artist(legend)
        else:
            ax.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.7)
        
        ax.set_title(f"{method.upper()} visualization of embeddings")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        
        # Save if requested
        if save_path:
            save_path = Path(save_path)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        
        return fig


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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract embeddings from DAE-KAN-Attention model")
    
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        help="Name of the dataset to process"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="embeddings.csv",
        help="Path to save embeddings CSV (default: embeddings.csv)"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=64,
        help="Batch size (default: 64)"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        help="Device to run on (default: cuda if available, else cpu)"
    )
    
    parser.add_argument(
        "--num-workers", 
        type=int, 
        default=4,
        help="Number of workers for data loading (default: 4)"
    )
    
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Visualize embeddings using t-SNE"
    )
    
    return parser.parse_args()


def main():
    """Main function to extract embeddings from command line."""
    args = parse_args()
    
    try:
        # Create dataloader
        dataloader = create_dataloader(
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # Initialize embedding extractor
        extractor = DAEEmbeddingExtractor(
            checkpoint_path=args.checkpoint,
            device=args.device,
            batch_size=args.batch_size
        )
        
        # Process dataset
        embeddings, labels = extractor.process_dataset(
            dataloader=dataloader,
            save_path=args.output
        )
        
        # Visualize if requested
        if args.visualize:
            fig = extractor.visualize_embeddings(
                embeddings=embeddings,
                labels=labels,
                save_path=f"{os.path.splitext(args.output)[0]}_viz.png"
            )
            plt.show()
        
        logger.info(f"Successfully extracted embeddings for {len(embeddings)} samples")
        logger.info(f"Embeddings shape: {embeddings.shape}")
        logger.info(f"Saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
