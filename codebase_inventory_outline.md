# Histopathology Codebase Inventory and Documentation Outline

## Project Structure Overview
- **Project Name**: Histopathology
- **Main Package**: `histopathology/`
- **Build System**: Poetry (pyproject.toml)
- **Development Environment**: Python >=3.9.16,<3.13

## Phase-Based Directory Logic

### Phase 2 (`histopathology/models/Phase2/`)
- **Purpose**: MLP-based embedding classification and clustering experiments
- **Key Files**:
  - `mlp_embeddings.py` - PyTorch Lightning MLP for DAE embedding classification
  - `Clustering.py` - Clustering evaluation utilities 
  - `ClusteringDAE.ipynb` & `Clustering.ipynb` - Clustering analysis notebooks
  - `trained_model/` - Model checkpoints (MLP, KAN subdirectories)
  - `lightning_logs/` - PyTorch Lightning training logs
  - `wandb/` - Weights & Biases experiment tracking logs

## Implemented Model Classes and Locations

### 1. Restricted Boltzmann Machine
- **Location**: `histopathology/models/RBM/rbm.py`
- **Class**: `RestrictedBoltzmanMachine`
- **Features**: Visible/hidden node transformations, Gibbs sampling, free energy computation

### 2. Autoencoders
- **Base Location**: `histopathology/models/Autoencoders/`
- **Vanilla VAE**: `vanilla_vae.py` (empty file)
- **Deep Embedded Clustering**: `DeepEmbeddedClustering/`
  - `ptsdae/` - PyTorch Stacked Denoising Autoencoders
    - `dae.py`, `sdae.py`, `model.py` - DAE implementations
  - `ptdec/` - PyTorch Deep Embedded Clustering  
    - `dec.py`, `cluster.py`, `model.py` - DEC implementations
  - Training scripts: `mnist.py`, `mnist_dec.py`, `test.py`

### 3. Variational Autoencoders
- **Location**: `histopathology/models/VariationalAutoEncoder/`
- **VaDE**: `VaDE/VaDE.py`, `VaDE/training.py` - Variational Deep Embedding
- **MFCVAE**: `mfcvae/` - Multi-Faceted Convolutional VAE
  - Core model: `mfcvae.py`, `conv_vlae.py`, `models_fc.py`
  - Training: `train.py` (extensive argparse CLI)
  - Evaluation: `eval_*.py` scripts
  - Configuration: `configs/` directory with YAML/JSON configs
  - Shell scripts: `shell_scripts/` for different datasets

### 4. Attention-Based Models
- **Location**: `histopathology/models/AttentionBottleNeckedDAE/`
- **Main Models**: 
  - `models/model.py`, `models/model_rev2.py` - DAE with KAN and Attention
  - `models/dae_embedding.py` - Embedding extraction utilities
- **Attention Mechanisms**: `models/attention_mechanisms/`
  - CBAM, BAM, SE, ECA, Coordinate Attention, etc.
- **KAN Integration**:
  - `Convolutional-KANs/` - Convolutional Kolmogorov-Arnold Networks
  - `FCN-KAN/` - Fully Connected KAN implementations
- **External Dependencies**: `pytorch-attention/` - Comprehensive attention library
- **Training**: `models/pl_training.py`, `models/pl_training_pretrained.py`

### 5. Phase 2 MLP Models
- **Location**: `histopathology/models/Phase2/mlp_embeddings.py`
- **Class**: `MLP` (PyTorch Lightning Module)
- **Features**: Multi-layer perceptron for DAE embedding classification
- **Metrics**: Accuracy, F1-score tracking
- **Integration**: Wandb logging, model checkpointing

## Data Loading Utilities for HistoHepar Dataset

### Raw Data Processing
- **Location**: `histopathology/data/raw/data_hepar.py`
- **Function**: `get_directory_levels()`, `write_file_info_to_csv()`
- **Purpose**: Extracts directory structure and treatment metadata from HistoHepar dataset
- **Output**: CSV files with treatment types and image paths

### Interim Data Processing  
- **Location**: `histopathology/data/interim/data_hepar.py`
- **Purpose**: Intermediate processing for HistoHepar dataset

### Data Pipeline Utilities
- **Location**: `histopathology/src/data_pipeline/`
- **Key Files**:
  - `make_dataset.py` - Patch generation from whole slide images
    - `generate_patches()` - Creates 128x128 patches with patchify
    - `parallelPatching()` - Multiprocessing for batch patch generation
  - `preprocessing.py` - Stain normalization and preprocessing
  - `convertTIF2PNG.py` - Format conversion utilities
  - `process_pannuke.py` - PANnuke dataset processing

### Data Structure
```
histopathology/data/
├── raw/
│   ├── HistoHepar/ - Raw histopathology images
│   ├── PANnuke/ - Nuclei segmentation dataset  
│   └── HeparUnified/ - Unified hepatic dataset
├── interim/
│   ├── slides/ - Original slide images (HFD10X, HFD20X, HFD40X, ND200X, ND400X)
│   ├── tiles/ - Generated patches/tiles
│   ├── tiles_preproc/ - Preprocessed tiles
│   └── HeparUnifiedPNG/ - PNG converted images
└── processed/
    ├── DAE_Embeddings.csv - Deep autoencoder embeddings
    ├── mlp_embeddings.csv - MLP-generated embeddings
    └── train.csv, test.csv - Train/test splits
```

## Available Training/Evaluation Scripts and CLI Arguments

### 1. MFCVAE Training (`models/VariationalAutoEncoder/mfcvae/train.py`)
**Comprehensive CLI with 50+ arguments:**
- **Core**: `--device`, `--dataset`, `--model_type`
- **Architecture**: `--J_n_mixtures`, `--z_j_dim_list`, `--n_clusters_j_list`
- **Training**: `--n_epochs`, `--batch_size`, `--init_lr`, `--seed`
- **Progressive Training**: `--do_progressive_training`, `--n_epochs_per_progressive_step`
- **Dataset Configs**: `--factors_variation_dict`, `--factors_label_list` (for 3DShapes)
- **Model Configs**: `--encode_layer_dims`, `--decode_layer_dims`, `--activation`
- **Evaluation**: Shell scripts in `shell_scripts/` for different datasets

### 2. Attention-DAE Training (`models/AttentionBottleNeckedDAE/models/pl_training.py`)
**PyTorch Lightning-based:**
- Fixed hyperparameters (learning rate: 0.01, batch size: 12)
- Wandb integration for experiment tracking
- Automatic learning rate scheduling and early stopping
- Model checkpointing based on validation loss

### 3. Phase 2 MLP Training (`models/Phase2/mlp_embeddings.py`)
**Embedded training script:**
- Loads DAE embeddings from CSV
- Fixed architecture: 4096 → 2048 → 1024 → 1024 → 14 classes
- PyTorch Lightning with Wandb logging
- Stratified train/validation split (90/10)

### 4. Deep Embedded Clustering
**MNIST Examples**: `models/Autoencoders/DeepEmbeddedClustering/`
- `mnist.py` - Stacked DAE pretraining
- `mnist_dec.py` - Deep embedded clustering
- Configurable architectures and hyperparameters

### 5. Evaluation Scripts
**MFCVAE Evaluation**:
- `eval_sample_generation.py` - Generate samples from learned clusters
- `eval_top10_cluster_examples.py` - Visualize cluster representatives  
- `eval_compositionality.py` - Test compositional generation

## Dataset Information

### HistoHepar Dataset Structure
- **Treatment Types**: HFD (High Fat Diet), ND (Normal Diet)
- **Magnifications**: 10X, 20X, 40X, 200X, 400X
- **Treatment Subtypes**: Various concentrations (A0, A5, A10, A15, A20, A25, etc.)
- **Total Images**: ~21,168 tiled images
- **Format**: TIFF → PNG conversion pipeline
- **Patch Size**: 128x128 pixels

### Additional Datasets
- **PANnuke**: Multi-organ nuclei segmentation (19 tissue types)
- **3DShapes**: Synthetic dataset for disentanglement evaluation
- **MNIST/SVHN**: Standard benchmarks for autoencoder validation

## Notebook Documentation

### Analysis Notebooks
1. **`1.0-drm-datainspection.ipynb`** - Initial data exploration and statistics
2. **`2.0-drm-dataprep.ipynb`** - Data preprocessing and preparation
3. **`3.0-drm-DACtraining.ipynb`** - Deep autoencoder training
4. **`4.0-drm-VisualizationAndAnalysis.ipynb`** - Results visualization and analysis
5. **`Phase2/Clustering.ipynb`** - Clustering analysis and evaluation
6. **`Phase2/ClusteringDAE.ipynb`** - DAE-based clustering experiments

### Utility Notebooks
- **`test.ipynb`** - Testing and debugging
- **`plot_segmentations.ipynb`** - Segmentation visualization

## Dependencies and Environment

### Core Dependencies (from pyproject.toml)
- **Deep Learning**: PyTorch, TorchVision, PyTorch Lightning, MONAI
- **Computer Vision**: OpenCV, scikit-image, Pillow, openslide-python
- **Scientific Computing**: NumPy, scikit-learn, scipy, statsmodels
- **Visualization**: Matplotlib, Seaborn, Plotly, Altair
- **Data Processing**: Pandas, Polars, h5py, PyArrow
- **Experiment Tracking**: Wandb, TensorBoard
- **Medical Imaging**: DICOM support, ITK, SimpleITK
- **Specialized**: KAN networks (pykan), attention mechanisms, patchify

### Development Tools
- **Package Management**: Poetry
- **Data Versioning**: DVC
- **Notebooks**: Jupyter, JupyterLab
- **GPU Support**: CUDA, CuCIM for accelerated image processing

## Key Findings Summary

1. **Phase-based Organization**: Currently only Phase2 exists, focusing on MLP classification of DAE embeddings
2. **Comprehensive Model Library**: RBM, various autoencoders, attention mechanisms, KAN integration
3. **Extensive MFCVAE Framework**: Most sophisticated model with full CLI, evaluation suite, and configuration system
4. **HistoHepar-Specific Pipeline**: Custom data loading, patching, and preprocessing for hepatic histopathology
5. **Experiment Tracking**: Heavy integration with Wandb and PyTorch Lightning for reproducible ML
6. **Multi-Modal Support**: Handles various medical imaging formats and external datasets

