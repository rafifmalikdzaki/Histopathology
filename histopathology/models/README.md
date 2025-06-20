# Models Directory

This directory contains all deep learning models and components used in the histopathology image analysis project. The structure has been reorganized for better maintainability and clarity.

## Directory Structure

```
models/
├── autoencoders/           # Autoencoder model implementations
├── components/             # Reusable model components
├── experiments/            # Training and evaluation scripts
├── utils/                  # Utility functions and helpers
└── README.md              # This file
```

## Autoencoders

### DAE KAN Attention (`autoencoders/dae_kan_attention/`)
A sophisticated Deep Autoencoder combining Kolmogorov-Arnold Networks (KAN) with attention mechanisms.

**Key Features:**
- Combines KAN layers with traditional CNNs
- Incorporates BAM (Bottleneck Attention Module) and ECA (Efficient Channel Attention)
- Supports residual connections for better gradient flow
- Designed for histopathology image reconstruction and feature extraction

**Main Components:**
- `model.py`: Complete DAE_KAN_Attention architecture
- `KANConv.py`, `KANLinear.py`: KAN layer implementations
- `pl_training_pretrained.py`: PyTorch Lightning training script
- `histopathology_dataset.py`: Dataset handling for histopathology images
- `dae_embedding.py`: Feature extraction utilities

**Usage:**
```python
from histopathology.models.autoencoders.dae_kan_attention.model import DAE_KAN_Attention

model = DAE_KAN_Attention(device='cuda')
encoded, decoded, latent = model(input_images)
```

### Deep Embedded Clustering (`autoencoders/deep_embedded_clustering/`)
Implementation of deep embedded clustering for unsupervised learning on histopathology data.

**Components:**
- `ptdec/`: PyTorch implementation of Deep Embedded Clustering
- `ptsdae/`: Stacked Denoising Autoencoder components

### Vanilla VAE (`autoencoders/vanilla_vae/`)
Standard Variational Autoencoder implementation.

### Variational (`autoencoders/variational/`)
VaDE (Variational Deep Embedding) for joint representation learning and clustering.

## Components

### Attention Mechanisms (`components/attention_mechanisms/`)
Collection of attention mechanisms for enhancing model performance:

- `bam.py`: Bottleneck Attention Module
- `cbam.py`: Convolutional Block Attention Module
- `eca.py`: Efficient Channel Attention
- `se_module.py`: Squeeze-and-Excitation
- `simam.py`: SimAM attention
- `coordatten.py`: Coordinate Attention
- And more...

### KAN Components (`components/kan/`)
Kolmogorov-Arnold Network implementations:

- `kan_layer.py`: Basic KAN layer implementation
- `KANConv.py`: Convolutional KAN layers
- `KANLinear.py`: Linear KAN layers

## Experiments

Training and evaluation scripts:
- `Clustering.py`: Clustering experiments
- `mlp_embeddings.py`: MLP-based embedding experiments

## Utilities

Helper functions and utility classes:
- `rbm.py`: Restricted Boltzmann Machine implementation

## Installation and Setup

1. Ensure you have the required dependencies installed (see main project requirements)
2. The models use relative imports, so import from the package root:

```python
from histopathology.models.autoencoders.dae_kan_attention import DAE_KAN_Attention
from histopathology.models.components.attention_mechanisms.bam import BAM
```

## Model Training

### DAE KAN Attention Training

```python
from histopathology.models.autoencoders.dae_kan_attention.pl_training_pretrained import HistoDAE

# Initialize trainer
trainer = HistoDAE(learning_rate=1e-4, batch_size=32)

# Train model
trainer.fit()
```

### Using Pre-trained Models

Pre-trained models should be stored in the appropriate subdirectories. Load them using:

```python
model = DAE_KAN_Attention.load_from_checkpoint('path/to/checkpoint.ckpt')
```

## Key Features by Model

| Model | Key Features | Use Case |
|-------|-------------|----------|
| DAE KAN Attention | KAN + Attention + Residuals | Feature extraction, reconstruction |
| Deep Embedded Clustering | Unsupervised clustering | Data exploration, segmentation |
| Vanilla VAE | Standard VAE | Baseline comparison |
| VaDE | Joint clustering + representation | Structured data analysis |

## Performance Considerations

- **GPU Memory**: DAE KAN Attention requires significant GPU memory due to KAN layers
- **Training Time**: KAN-based models train slower than standard CNNs
- **Batch Size**: Adjust batch sizes based on available GPU memory

## Contributing

When adding new models:
1. Place them in the appropriate subdirectory
2. Include proper documentation and docstrings
3. Add __init__.py files for new packages
4. Update this README with model descriptions
5. Ensure proper relative imports

## Notes

- The old models directory has been moved to `models_old/` for reference
- All nested git repositories have been flattened to avoid conflicts
- Experiment artifacts (wandb logs, checkpoints) should be stored outside the source tree

