# DAE KAN Attention Model

A sophisticated Deep Autoencoder combining Kolmogorov-Arnold Networks (KAN) with attention mechanisms for histopathology image analysis.

## Architecture Overview

The DAE_KAN_Attention model consists of three main components:

1. **Autoencoder_Encoder**: Initial encoding with residual connections
2. **Autoencoder_BottleNeck**: Feature compression with BAM attention
3. **Autoencoder_Decoder**: Reconstruction with KAN layers

## Key Features

- **KAN Integration**: Uses Kolmogorov-Arnold Networks for enhanced expressivity
- **Attention Mechanisms**: BAM (Bottleneck Attention Module) and ECA (Efficient Channel Attention)
- **Residual Connections**: Improved gradient flow and training stability
- **Multi-scale Processing**: Progressive encoding/decoding at different resolutions

## Files Description

### Core Model Files
- `model.py`: Main model architecture (DAE_KAN_Attention class)
- `KANConv.py`: Convolutional KAN layer implementation
- `KANLinear.py`: Linear KAN layer implementation

### Training and Data
- `pl_training_pretrained.py`: PyTorch Lightning training wrapper
- `pl_training.py`: Alternative training script
- `histopathology_dataset.py`: Dataset loader for histopathology images
- `dae_embedding.py`: Feature extraction utilities

### Additional Components
- `model_rev2.py`: Alternative model architecture
- `convolution.py`: Additional convolutional utilities

## Usage

### Basic Model Usage

```python
import torch
from model import DAE_KAN_Attention

# Initialize model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DAE_KAN_Attention(device=device).to(device)

# Forward pass
input_images = torch.randn(batch_size, 3, 128, 128).to(device)
encoded, decoded, latent = model(input_images)

print(f"Input shape: {input_images.shape}")
print(f"Encoded shape: {encoded.shape}")
print(f"Decoded shape: {decoded.shape}")
print(f"Latent shape: {latent.shape}")
```

### Training with PyTorch Lightning

```python
from pl_training_pretrained import HistoDAE
import pytorch_lightning as pl

# Initialize model
model = HistoDAE(
    learning_rate=1e-4,
    batch_size=32,
    max_epochs=100
)

# Create trainer
trainer = pl.Trainer(
    max_epochs=100,
    gpus=1 if torch.cuda.is_available() else 0,
    precision=16  # Use mixed precision for memory efficiency
)

# Train
trainer.fit(model)
```

### Feature Extraction

```python
from dae_embedding import extract_features

# Extract features from trained model
features = extract_features(model, dataloader, device)
```

## Model Architecture Details

### Encoder Path
1. **Initial Convolution**: 3 → 384 channels, stride=2
2. **Second Layer**: 384 → 128 channels, stride=2  
3. **Third Layer**: 128 → 64 channels, stride=2
4. **KAN Processing**: Applied to 64-channel feature maps
5. **ECA Attention**: Channel attention on encoded features

### Bottleneck
1. **Compression**: 384 → 384 → 16 channels with stride=2 each
2. **BAM Attention**: Applied at both 384 and 16 channel stages
3. **Latent Representation**: 16-channel feature maps

### Decoder Path
1. **Initial Upsampling**: 16 → 128 channels
2. **Progressive Reconstruction**: Through 384 and 3 channels
3. **Residual Addition**: Encoder features added to decoder
4. **Final KAN**: Applied for final reconstruction refinement

## Dependencies

- PyTorch >= 1.9.0
- PyTorch Lightning >= 1.5.0
- torchvision
- numpy
- PIL (for image processing)

## Performance Notes

- **Memory Requirements**: ~8GB GPU memory for batch_size=32, 128x128 images
- **Training Time**: ~2-3x slower than standard CNNs due to KAN layers
- **Convergence**: Typically converges within 50-100 epochs

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch_size or image resolution
2. **Import Errors**: Ensure proper relative imports from package root
3. **Slow Training**: KAN layers are computationally intensive - consider mixed precision

### Recommended Settings

- **Batch Size**: 16-32 (depending on GPU memory)
- **Learning Rate**: 1e-4 to 1e-3
- **Image Size**: 128x128 or 256x256
- **Mixed Precision**: Recommended for memory efficiency

## Citation

If you use this model in your research, please cite:
```
[Add appropriate citation when paper is published]
```

