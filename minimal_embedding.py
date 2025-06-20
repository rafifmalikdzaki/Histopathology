#!/usr/bin/env python3
# minimal_embedding.py - Extract embeddings using minimal model

import torch
import torch.utils.data
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import argparse
from pathlib import Path

from minimal_dae_model import load_minimal_model
from histopathology.models.autoencoders.dae_kan_attention.histopathology_dataset import create_dataset, ImageDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def extract_embeddings(
    checkpoint_path: str,
    dataset_name: str,
    output_path: str = "embeddings.csv",
    batch_size: int = 192,
    device: str = None
):
    """
    Extract embeddings using minimal model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        dataset_name: Name of dataset to process
        output_path: Path to save embeddings CSV
        batch_size: Batch size for processing
        device: Device to use ('cuda' or 'cpu')
    """
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    try:
        # Load minimal model
        model = load_minimal_model(checkpoint_path, device)
        model.eval()
        
        # Create dataset
        logger.info(f"Creating dataset: {dataset_name}")
        data = create_dataset(dataset_name)
        dataset = ImageDataset(*data, device=None)
        
        logger.info(f"Creating dataloader with batch size {batch_size}")
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Extract embeddings
        logger.info("Extracting embeddings")
        embeddings = []
        labels = []
        
        with torch.no_grad():
            for data, label in tqdm(dataloader, desc="Processing batches"):
                # Move data to device
                data = data.to(device)
                
                # Forward pass
                _, _, z = model(data)
                
                # Flatten and store embeddings
                z = z.view(z.size(0), -1)
                embeddings.append(z.cpu().numpy())
                labels.append(label.cpu().numpy())
                
                # Clear GPU memory
                if device == 'cuda':
                    torch.cuda.empty_cache()
        
        # Concatenate results
        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        # Save to CSV
        logger.info(f"Saving embeddings to {output_path}")
        df = pd.DataFrame(embeddings)
        df['label'] = labels
        df.to_csv(output_path, index=False)
        
        logger.info(f"Embedding extraction complete: {embeddings.shape}")
        return embeddings, labels
        
    except Exception as e:
        logger.error(f"Error extracting embeddings: {e}")
        raise

def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Extract embeddings using minimal DAE model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--output", default="embeddings.csv", help="Output CSV path")
    parser.add_argument("--batch-size", type=int, default=192, help="Batch size")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    try:
        extract_embeddings(
            checkpoint_path=args.checkpoint,
            dataset_name=args.dataset,
            output_path=args.output,
            batch_size=args.batch_size,
            device=args.device
        )
        return 0
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
