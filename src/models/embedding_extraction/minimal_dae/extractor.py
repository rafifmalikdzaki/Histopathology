import logging
import os
import json
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm

from .model import MinimalDAEModel, load_minimal_model
from ..base import BaseEmbeddingExtractor

# Configure logger
logger = logging.getLogger(__name__)

class MinimalDAEExtractor(BaseEmbeddingExtractor):
    """
    Embedding extractor for the Minimal DAE Model.
    
    Extracts embeddings using the Minimal DAE architecture with encoded components,
    focusing on batch processing and efficient data handling.
    
    Example:
        >>> extractor = MinimalDAEExtractor('path/to/checkpoint.pth', device='cuda')
        >>> extractor.load_model()
        >>> # Extract embeddings from a single batch
        >>> batch = torch.randn(32, 3, 224, 224)
        >>> embeddings = extractor.process_batch(batch)
        >>> # Or process an entire dataset
        >>> extractor.process_dataset(dataloader, save_path='embeddings.npz')
    """
    
    def __init__(
        self, 
        model_path: Union[str, Path], 
        device: str = 'cuda',
        embedding_dim: int = 4096,
        cache_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the MinimalDAE embedding extractor.
        
        Args:
            model_path: Path to the DAE model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
            embedding_dim: Expected dimension of the output embeddings
            cache_dir: Directory to cache intermediate results (defaults to same directory as model_path)
        
        Example:
            >>> extractor = MinimalDAEExtractor('/path/to/checkpoint.pth')
        """
        super().__init__(model_path, device)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.embedding_dim = embedding_dim
        self.cache_dir = Path(cache_dir) if cache_dir else Path(model_path).parent / "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self._metadata = {}

    def load_model(self) -> None:
        """
        Load the minimal DAE model weights from a checkpoint.
        
        This method loads the MinimalDAEModel using the checkpoint path provided
        during initialization and sets it to evaluation mode.
        
        Raises:
            FileNotFoundError: If the model checkpoint file doesn't exist
            RuntimeError: If there's an error loading the model
            
        Example:
            >>> extractor = MinimalDAEExtractor('path/to/checkpoint.pth')
            >>> extractor.load_model()
        """
        try:
            self.logger.info(f"Loading model from {self.model_path}")
            self.model = load_minimal_model(self.model_path, self.device)
            self.model.eval()  # Ensure model is in evaluation mode
            
            # Store model metadata
            self._metadata['model_type'] = 'MinimalDAE'
            self._metadata['checkpoint_path'] = str(self.model_path)
            self._metadata['device'] = self.device
            
            self.logger.info("Model loaded successfully.")
        except FileNotFoundError as e:
            self.logger.error(f"Model file not found: {e}")
            raise
        except RuntimeError as e:
            self.logger.error(f"Failed to load the model: {e}")
            raise

    def extract_embeddings(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from input data using the loaded DAE model.
        
        Args:
            inputs: Batch of input data as tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Extracted embeddings as tensor of shape (batch_size, embedding_dim)
            
        Raises:
            RuntimeError: If model is not loaded or extraction fails
            ValueError: If input shape is invalid
            
        Example:
            >>> inputs = torch.randn(32, 3, 224, 224)
            >>> embeddings = extractor.extract_embeddings(inputs)
            >>> print(embeddings.shape)
            torch.Size([32, 4096])
        """
        if self.model is None:
            self.logger.error("Model not loaded. Call load_model() first.")
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        # Validate input shape
        self._validate_input(inputs)
        
        self.model.eval()
        with torch.no_grad():
            try:
                embeddings = self.model.extract_embeddings(inputs)
                
                # Validate output shape
                if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
                    self.logger.warning(
                        f"Unexpected embedding shape: {embeddings.shape}, "
                        f"expected second dimension to be {self.embedding_dim}"
                    )
                
                return embeddings
            except Exception as e:
                self.logger.error(f"Failed to extract embeddings: {e}")
                raise RuntimeError(f"Failed to extract embeddings: {e}")
    
    def process_batch(
        self, 
        batch: torch.Tensor,
        return_numpy: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Process a single batch of inputs and extract embeddings.
        
        This method handles moving data to the correct device, validating input shapes,
        and extracting embeddings in a memory-efficient way.
        
        Args:
            batch: A batch of input data as tensor of shape (batch_size, channels, height, width)
            return_numpy: Whether to return embeddings as numpy array (True) or torch tensor (False)
            
        Returns:
            Embeddings as numpy array or torch tensor
            
        Raises:
            ValueError: If input batch has invalid shape
            RuntimeError: If model is not loaded or extraction fails
            
        Example:
            >>> batch = torch.randn(32, 3, 224, 224)
            >>> embeddings = extractor.process_batch(batch)
            >>> print(embeddings.shape)
            (32, 4096)
        """
        # Validate input data
        self._validate_input(batch)
        
        # Ensure model is loaded
        if self.model is None:
            self.logger.info("Model not loaded, loading now...")
            self.load_model()
            
        try:
            # Move batch to appropriate device
            batch = batch.to(self.device)
            
            # Extract embeddings
            with torch.no_grad():
                embeddings = self.extract_embeddings(batch)
                
            # Convert to numpy if requested
            if return_numpy:
                return embeddings.cpu().numpy()
            else:
                return embeddings.cpu()
                
        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")
            raise RuntimeError(f"Error processing batch: {e}")
    
    def process_dataset(
        self,
        dataloader: DataLoader,
        save_path: Union[str, Path] = None,
        show_progress: bool = True,
        checkpoint_freq: int = 50,
        resume_from_checkpoint: bool = True,
        metadata: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process an entire dataset and extract embeddings.
        
        This method handles memory-efficient batch processing with checkpointing and
        resuming capability.
        
        Args:
            dataloader: PyTorch DataLoader to iterate over batches.
            save_path: Optional path to save embeddings.
            show_progress: Whether to show a progress bar.
            checkpoint_freq: Frequency to save checkpoints.
            resume_from_checkpoint: Whether to try resuming from last checkpoint if available.
            metadata: Optional metadata to save with embeddings.
            
        Returns:
            Tuple of (embeddings, labels) arrays.
            
        Raises:
            RuntimeError: If processing fails.
            
        Example:
            >>> dataset = torch.utils.data.TensorDataset(images, labels)
            >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
            >>> embeddings, labels = extractor.process_dataset(dataloader, save_path='embeddings.npz')
        """
        # Initialize empty lists for storing results
        embeddings_list = []
        labels_list = []
        
        # Ensure model is loaded
        if self.model is None:
            self.logger.info("Model not loaded, loading now...")
            self.load_model()
        
        # Set up checkpoint path
        checkpoint_base = None
        last_processed_batch = -1
        
        if save_path:
            save_path = Path(save_path)
            checkpoint_base = save_path.parent / f"{save_path.stem}_checkpoint"
            
            # Try to resume from checkpoint if requested
            if resume_from_checkpoint:
                last_processed_batch = self._find_latest_checkpoint(checkpoint_base)
                if last_processed_batch >= 0:
                    try:
                        # Load embeddings and labels from checkpoint
                        checkpoint_path = f"{checkpoint_base}_{last_processed_batch}.npz"
                        self.logger.info(f"Resuming from checkpoint {checkpoint_path}")
                        loaded_data = np.load(checkpoint_path)
                        embeddings_list = [loaded_data['embeddings']]
                        labels_list = [loaded_data['labels']]
                    except Exception as e:
                        self.logger.warning(f"Failed to resume from checkpoint: {e}")
                        last_processed_batch = -1
                        embeddings_list = []
                        labels_list = []
        
        # Create progress bar
        iterator = tqdm(dataloader, desc="Extracting embeddings") if show_progress else dataloader
        
        # Create metadata for the extraction process
        self._metadata.update({
            'total_batches': len(dataloader),
            'batch_size': dataloader.batch_size,
            'extraction_time': None,  # Will be updated at the end
            'dataset_size': len(dataloader.dataset) if hasattr(dataloader, 'dataset') else None,
        })
        
        # Update with user-provided metadata
        if metadata:
            self._metadata.update(metadata)
        
        # Process batches
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        
        try:
            # Skip batches that were already processed in the checkpoint
            for batch_idx, (inputs, labels) in enumerate(iterator):
                if batch_idx <= last_processed_batch:
                    continue
                
                # Process batch
                self.logger.debug(f"Processing batch {batch_idx}/{len(dataloader)}")
                embeddings = self.process_batch(inputs)
                embeddings_list.append(embeddings)
                labels_list.append(labels.cpu().numpy() if torch.is_tensor(labels) else labels)
                
                # Save checkpoint if needed
                if checkpoint_base is not None and batch_idx % checkpoint_freq == 0 and batch_idx > 0:
                    self.logger.info(f"Saving checkpoint at batch {batch_idx}/{len(dataloader)}")
                    try:
                        intermediate_embeds = np.concatenate(embeddings_list, axis=0)
                        intermediate_labels = np.concatenate(labels_list, axis=0)
                        checkpoint_save_path = f"{checkpoint_base}_{batch_idx}.npz"
                        np.savez(
                            checkpoint_save_path, 
                            embeddings=intermediate_embeds, 
                            labels=intermediate_labels
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to save checkpoint: {e}")
                
                # Update progress bar with memory usage if available
                if show_progress and torch.cuda.is_available():
                    mem_allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
                    mem_reserved = torch.cuda.memory_reserved() / 1024**3
                    iterator.set_postfix({
                        'memory_allocated': f"{mem_allocated:.2f} GB",
                        'memory_reserved': f"{mem_reserved:.2f} GB"
                    })
            
            # Record end time
            end_time.record()
            torch.cuda.synchronize()
            extraction_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
            self._metadata['extraction_time'] = extraction_time
            
            # Concatenate all results
            final_embeds = np.concatenate(embeddings_list, axis=0)
            final_labels = np.concatenate(labels_list, axis=0)
            
            # Save final embeddings if requested
            if save_path:
                self.logger.info(f"Saving final embeddings to {save_path}...")
                self.save_embeddings(final_embeds, final_labels, save_path, metadata=self._metadata)
                
                # Clean up checkpoints
                self._cleanup_checkpoints(checkpoint_base)
            
            return final_embeds, final_labels
            
        except KeyboardInterrupt:
            self.logger.warning("Processing interrupted by user. Saving intermediate results...")
            if embeddings_list and save_path:
                try:
                    intermediate_embeds = np.concatenate(embeddings_list, axis=0)
                    intermediate_labels = np.concatenate(labels_list, axis=0)
                    interrupt_save_path = f"{save_path.parent}/{save_path.stem}_interrupted.npz"
                    self.save_embeddings(intermediate_embeds, intermediate_labels, interrupt_save_path)
                    self.logger.info(f"Intermediate results saved to {interrupt_save_path}")
                except Exception as e:
                    self.logger.error(f"Failed to save intermediate results: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error processing dataset: {e}")
            raise RuntimeError(f"Error processing dataset: {e}")
            
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        save_path: Union[str, Path],
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Save embeddings and labels to a file.
        
        Args:
            embeddings: Numpy array of embeddings.
            labels: Numpy array of corresponding labels.
            save_path: Path to save the embeddings to.
            metadata: Optional metadata to save with embeddings.
            
        Raises:
            IOError: If there's an error saving the embeddings.
            
        Example:
            >>> embeddings = np.random.randn(100, 4096)
            >>> labels = np.random.randint(0, 10, size=(100,))
            >>> extractor.save_embeddings(embeddings, labels, 'embeddings.npz')
        """
        save_path = Path(save_path)
        self.logger.info(f"Saving embeddings (shape: {embeddings.shape}) to {save_path}")
        
        # Get additional metadata
        combined_metadata = {
            'embedding_dim': embeddings.shape[1],
            'num_samples': embeddings.shape[0],
            'save_timestamp': pd.Timestamp.now().isoformat(),
            'model_checkpoint': str(self.model_path)
        }
        
        # Add user metadata if provided
        if metadata:
            combined_metadata.update(metadata)
        
        try:
            # Save in appropriate format based on file extension
            if save_path.suffix == '.npz':
                # Save as NumPy compressed format
                np.savez(
                    save_path, 
                    embeddings=embeddings, 
                    labels=labels, 
                    metadata=json.dumps(combined_metadata)
                )
                
            elif save_path.suffix == '.csv':
                # Save as CSV with metadata in first row as comment
                import pandas as pd
                df = pd.DataFrame(embeddings)
                df['label'] = labels
                
                # Add metadata as commented header
                with open(save_path, 'w') as f:
                    f.write(f"# {json.dumps(combined_metadata)}\n")
                
                # Append DataFrame without index
                df.to_csv(save_path, index=False, mode='a')
                
            else:
                # Default to NPZ format for unknown extensions
                self.logger.warning(
                    f"Unknown file extension: {save_path.suffix}, saving as .npz"
                )
                np.savez(
                    save_path, 
                    embeddings=embeddings, 
                    labels=labels,
                    metadata=json.dumps(combined_metadata)
                )
                
            self.logger.info(f"Successfully saved embeddings to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving embeddings: {e}")
            raise IOError(f"Failed to save embeddings: {e}")
            
    def load_embeddings(self, load_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load embeddings and labels from a file.
        
        Args:
            load_path: Path to the embeddings file.
            
        Returns:
            Tuple of (embeddings, labels, metadata) where metadata is a dictionary
            
        Raises:
            FileNotFoundError: If the embeddings file doesn't exist.
            ValueError: If the file format is invalid.
            
        Example:
            >>> embeddings, labels, metadata = extractor.load_embeddings('embeddings.npz')
            >>> print(f"Loaded {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")
        """
        load_path = Path(load_path)
        self.logger.info(f"Loading embeddings from {load_path}")
        
        if not load_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {load_path}")
        
        try:
            # Handle different file formats
            if load_path.suffix == '.npz':
                # Load from NumPy compressed format
                data = np.load(load_path, allow_pickle=True)
                embeddings = data['embeddings']
                labels = data['labels']
                
                # Try to load metadata if available
                metadata = {}
                if 'metadata' in data:
                    try:
                        metadata = json.loads(data['metadata'].item())
                    except:
                        self.logger.warning("Failed to parse metadata from NPZ file")
                
            elif load_path.suffix == '.csv':
                # Load from CSV, assuming label is the last column
                import pandas as pd
                
                # Check for metadata in first line
                metadata = {}
                with open(load_path, 'r') as f:
                    first_line = f.readline()
                    if first_line.startswith('#'):
                        try:
                            metadata = json.loads(first_line[1:].strip())
                        except:
                            self.logger.warning("Failed to parse metadata from CSV header")
                
                # Load the data
                df = pd.read_csv(load_path, comment='#')
                labels = df['label'].values
                embeddings = df.drop('label', axis=1).values
                
            else:
                raise ValueError(f"Unsupported file format: {load_path.suffix}")
            
            self.logger.info(
                f"Successfully loaded embeddings with shape {embeddings.shape} "
                f"and {len(labels)} labels"
            )
            
            return embeddings, labels, metadata
            
        except Exception as e:
            self.logger.error(f"Error loading embeddings: {e}")
            raise ValueError(f"Failed to load embeddings from {load_path}: {e}")
    
    def verify_embeddings(
        self, 
        embeddings: np.ndarray, 
        n_samples: int = 10,
        check_nan: bool = True,
        check_range: bool = True,
        check_clustering: bool = False
    ) -> Dict[str, Any]:
        """
        Verify the quality of extracted embeddings.
        
        This method performs basic quality checks on embeddings.
        
        Args:
            embeddings: The embedding matrix to verify
            n_samples: Number of samples to use for detailed checks
            check_nan: Whether to check for NaN values
            check_range: Whether to check for reasonable value ranges
            check_clustering: Whether to check basic clustering quality
            
        Returns:
            Dictionary with verification results
            
        Example:
            >>> embeddings = extractor.process_dataset(dataloader)[0]
            >>> results = extractor.verify_embeddings(embeddings)
            >>> if results['valid']:
            >>>     print("Embeddings look good!")
            >>> else:
            >>>     print(f"Issues with embeddings: {results['issues']}")
        """
        results = {
            'valid': True,
            'issues': [],
            'stats': {}
        }
        
        # Check for NaN values
        if check_nan:
            nan_count = np.isnan(embeddings).sum()
            if nan_count > 0:
                results['valid'] = False
                results['issues'].append(f"Found {nan_count} NaN values")
            results['stats']['nan_count'] = nan_count
        
        # Check for reasonable value ranges
        if check_range:
            min_val = embeddings.min()
            max_val = embeddings.max()
            mean_val = embeddings.mean()
            std_val = embeddings.std()
            
            results['stats'].update({
                'min': float(min_val),
                'max': float(max_val),
                'mean': float(mean_val),
                'std': float(std_val)
            })
            
            # Flag extremely large values that could indicate issues
            if abs(max_val) > 100 or abs(min_val) > 100:
                results['issues'].append(f"Extreme values detected: min={min_val}, max={max_val}")
                results['valid'] = False
            
            # Flag if all values are the same (constant embeddings)
            if std_val < 1e-6:
                results['issues'].append("Near-zero standard deviation indicates constant embeddings")
                results['valid'] = False
        
        # Basic clustering check if requested
        if check_clustering and embeddings.shape[0] > n_samples:
            try:
                from sklearn.cluster import KMeans
                from sklearn.metrics import silhouette_score
                
                # Sample subset for clustering analysis
                indices = np.random.choice(embeddings.shape[0], min(1000, embeddings.shape[0]), replace=False)
                sample = embeddings[indices]
                
                # Run K-means with a small number of clusters
                kmeans = KMeans(n_clusters=min(5, sample.shape[0] // 10), random_state=42)
                labels = kmeans.fit_predict(sample)
                
                # Calculate silhouette score
                if len(np.unique(labels)) > 1:  # Need at least 2 clusters
                    score = silhouette_score(sample, labels)
                    results['stats']['silhouette_score'] = float(score)
                    
                    # Extremely low score could indicate poor embedding quality
                    if score < 0.1:
                        results['issues'].append(f"Low clustering quality (silhouette={score:.3f})")
                        results['valid'] = results['valid'] and False
            except ImportError:
                self.logger.warning("sklearn not available for clustering check")
            except Exception as e:
                self.logger.warning(f"Clustering check failed: {e}")
        
        return results

    def _validate_input(self, inputs: torch.Tensor) -> None:
        """
        Validate the input tensor shape and values.
        
        Args:
            inputs: Input tensor to validate
            
        Raises:
            ValueError: If validation fails
        """
        # Check tensor type
        if not isinstance(inputs, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(inputs)}")
        
        # Check shape
        if inputs.ndim != 4:
            raise ValueError(f"Expected 4D tensor (B,C,H,W), got shape {inputs.shape}")
        
        # Check channels
        if inputs.shape[1] != 3:
            raise ValueError(f"Expected 3 channels, got {inputs.shape[1]}")
        
        # Basic range check for image data
        if inputs.min() < -10 or inputs.max() > 10:
            self.logger.warning(
                f"Input values outside expected range: min={inputs.min()}, max={inputs.max()}. "
                "Verify normalization."
            )
    
    def _find_latest_checkpoint(self, checkpoint_base: Path) -> int:
        """
        Find the latest checkpoint batch index.
        
        Args:
            checkpoint_base: Base path for checkpoints
            
        Returns:
            Index of the latest checkpoint, or -1 if none found
        """
        # Get list of all checkpoint files
        parent_dir = checkpoint_base.parent
        prefix = checkpoint_base.name
        
        if not parent_dir.exists():
            return -1
            
        checkpoint_files = list(parent_dir.glob(f"{prefix}_*.npz"))
        
        if not checkpoint_files:
            return -1
            
        # Extract batch indices
        batch_indices = []
        for f in checkpoint_files:
            try:
                batch_idx = int(f.stem.split('_')[-1])
                batch_indices.append(batch_idx)
            except (ValueError, IndexError):
                continue
                
        return max(batch_indices) if batch_indices else -1
    
    def _cleanup_checkpoints(self, checkpoint_base: Path) -> None:
        """
        Clean up intermediate checkpoint files after successful completion.
        
        Args:
            checkpoint_base: Base path for checkpoints
        """
        parent_dir = checkpoint_base.parent
        prefix = checkpoint_base.name
        
        if not parent_dir.exists():
            return
            
        checkpoint_files = list(parent_dir.glob(f"{prefix}_*.npz"))
        
        if not checkpoint_files:
            return
            
        self.logger.info(f"Cleaning up {len(checkpoint_files)} checkpoint files")
        
        for f in checkpoint_files:
            try:
                os.remove(f)
            except Exception as e:
                self.logger.warning(f"Failed to remove checkpoint file {f}: {e}")
    
    def cleanup(self) -> None:
        """
        Clean up resources used by the extractor.
        
        This method releases GPU memory and cleans up temporary files.
        
        Example:
            >>> extractor.cleanup()
        """
        self.logger.info("Cleaning up resources")
        
        # Release model from GPU memory
        if self.model is not None:
            self.model = self.model.cpu()
            del self.model
            self.model = None
        
        # Clean CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.logger.info("Resources cleaned up")
    
    def __enter__(self):
        """Context manager entry - enables 'with' statement usage."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.cleanup()
        return False  # Don't suppress exceptions
