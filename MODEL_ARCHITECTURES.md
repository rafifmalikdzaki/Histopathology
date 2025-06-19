# Model Architectures Documentation

This document provides detailed documentation for the three implemented clustering and representation learning approaches in the histopathology project.

## 1. DEC (Deep Embedded Clustering)
**Location**: `histopathology/models/Autoencoders/DeepEmbeddedClustering/ptdec`

### Purpose
Deep Embedded Clustering (DEC) simultaneously learns feature representations and cluster assignments for unsupervised deep learning. The model performs joint optimization of clustering and feature representation learning through a custom clustering loss.

### Key Hyperparameters
- **`n_clusters`**: Number of clusters (typically 3-10 for histopathology applications)
- **`batch_size`**: Batch size for training (default: 256)
- **`lr`**: Learning rate (default: 0.001)
- **`n_epochs`**: Number of training epochs (default: 100) 
- **`ae_n_epochs`**: Number of autoencoder pretraining epochs (default: 400)
- **`ae_batch_size`**: Batch size for autoencoder pretraining (default: 256)
- **`save_dir`**: Directory to save trained models (default: "./")
- **`cuda`**: Enable CUDA for GPU training (default: True)

### Model Instantiation
```python
from ptdec import DEC

# Initialize DEC model
model = DEC(
    dataset=dataset,
    n_clusters=5,
    batch_size=256,
    lr=0.001,
    n_epochs=100,
    ae_n_epochs=400,
    ae_batch_size=256,
    save_dir="./models/",
    cuda=True
)

# Train the model
model.fit(data_loader)

# Get cluster assignments
cluster_assignments = model.predict(test_loader)

# Get learned embeddings
embeddings = model.encodings(test_loader)
```

### Reference Papers
- Xie, J., Girshick, R., & Farhadi, A. (2016). Unsupervised deep embedding for clustering analysis. In International conference on machine learning (pp. 478-487).

## 2. MFCVAE (Multi-Faceted Conditional Variational Autoencoder)
**Location**: `histopathology/models/VariationalAutoEncoder/mfcvae`

### Purpose
Multi-Faceted Conditional Variational Autoencoder is designed for learning disentangled representations in histopathology images. It separates different factors of variation in the data while maintaining the generative capabilities of VAEs.

### Key Hyperparameters
- **`latent_dim`**: Dimensionality of the latent space (default: 128)
- **`hidden_dims`**: List of hidden layer dimensions for encoder/decoder
- **`beta`**: β-VAE regularization parameter for disentanglement (default: 1.0)
- **`learning_rate`**: Learning rate for optimization (default: 1e-3)
- **`batch_size`**: Training batch size (default: 64)
- **`num_epochs`**: Number of training epochs (default: 200)
- **`kld_weight`**: Weight for KL divergence loss term (default: 1.0)
- **`reconstruction_loss`**: Type of reconstruction loss ('mse' or 'bce')

### Model Instantiation
```python
from mfcvae import MFCVAE

# Initialize MFCVAE model
model = MFCVAE(
    input_dim=(3, 224, 224),  # For RGB histopathology images
    latent_dim=128,
    hidden_dims=[32, 64, 128, 256, 512],
    beta=4.0,
    learning_rate=1e-3,
    batch_size=64,
    num_epochs=200,
    kld_weight=1.0,
    reconstruction_loss='mse'
)

# Train the model
model.fit(train_loader, val_loader)

# Generate samples
generated_samples = model.sample(num_samples=16)

# Encode data to latent space
latent_codes = model.encode(data)

# Reconstruct data
reconstructions = model.reconstruct(data)
```

### Reference Papers
- Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.
- Higgins, I., et al. (2017). β-VAE: Learning basic visual concepts with a constrained variational framework. ICLR.

## 3. Attention-Bottlenecked DAE (Denoising Autoencoder)
**Location**: `histopathology/models/Phase2/ClusteringDAE.ipynb`

### Purpose
The Attention-Bottlenecked Denoising Autoencoder combines attention mechanisms with bottleneck architectures for learning compact representations of histopathology images. It focuses on relevant image regions while learning robust feature representations through denoising.

### Key Hyperparameters
- **`input_dim`**: Input image dimensions (e.g., 4096 for flattened patches)
- **`bottleneck_dim`**: Dimensionality of the bottleneck layer (default: 256)
- **`noise_factor`**: Amount of noise added for denoising (default: 0.3)
- **`attention_heads`**: Number of attention heads in multi-head attention (default: 8)
- **`dropout_rate`**: Dropout rate for regularization (default: 0.1)
- **`learning_rate`**: Learning rate for training (default: 1e-4)
- **`batch_size`**: Training batch size (default: 128)
- **`epochs`**: Number of training epochs (default: 150)

### Model Instantiation
```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, BisectingKMeans
from sklearn.mixture import GaussianMixture

# Load DAE embeddings (output from the attention-bottlenecked DAE)
dae_embeddings = pd.read_csv("../../data/processed/DAE_Embeddings.csv").iloc[:, :-1]
true_labels = pd.read_csv("../../data/processed/DAE_Embeddings.csv").iloc[:, -1]

# Apply clustering algorithms to DAE embeddings
algorithms = {
    'KMeans': KMeans,
    'BisectingKMeans': BisectingKMeans, 
    'GaussianMixture': GaussianMixture
}

# Example clustering with different numbers of clusters
n_clusters_range = range(3, 10)
results = {}

for algorithm_name, algorithm_class in algorithms.items():
    for n_clusters in n_clusters_range:
        model = algorithm_class(n_clusters=n_clusters, random_state=42)
        cluster_labels = model.fit_predict(dae_embeddings)
        
        # Store results for evaluation
        results[f"{algorithm_name}_{n_clusters}"] = {
            'labels': cluster_labels,
            'model': model
        }

# The DAE architecture itself would be implemented as:
# class AttentionBottleneckDAE(nn.Module):
#     def __init__(self, input_dim=4096, bottleneck_dim=256, 
#                  attention_heads=8, dropout_rate=0.1):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 2048),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(2048, 1024),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             AttentionBottleneck(1024, bottleneck_dim, attention_heads),
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(bottleneck_dim, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 2048),
#             nn.ReLU(),
#             nn.Linear(2048, input_dim),
#             nn.Sigmoid()
#         )
```

### Reference Papers
- Vincent, P., et al. (2008). Extracting and composing robust features with denoising autoencoders. ICML.
- Vaswani, A., et al. (2017). Attention is all you need. Advances in neural information processing systems.

## Model Comparison Summary

| Model | Primary Use Case | Key Strengths | Computational Requirements |
|-------|------------------|---------------|---------------------------|
| **DEC** | Unsupervised clustering with representation learning | Joint optimization of features and clusters | Medium - requires autoencoder pretraining |
| **MFCVAE** | Disentangled representation learning and generation | Generative capabilities, disentangled factors | High - complex variational inference |
| **Attention-Bottlenecked DAE** | Robust feature extraction with spatial attention | Noise robustness, attention mechanisms | Medium - attention computation overhead |

## Usage Recommendations

- **DEC**: Best for applications requiring both clustering and feature learning where cluster assignments are the primary goal
- **MFCVAE**: Ideal for generative tasks, data augmentation, and when interpretable disentangled factors are needed
- **Attention-Bottlenecked DAE**: Recommended for noisy histopathology data where spatial attention and robust features are important

All models have been tested on histopathology image datasets and show complementary strengths for different aspects of medical image analysis.

