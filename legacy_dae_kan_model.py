import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple, Union, Optional, Callable

# Import components from the original model
from histopathology.models.components.attention_mechanisms.bam import BAM
from histopathology.models.components.attention_mechanisms.eca import ECALayer
from histopathology.models.components.kan.kan_layer import KANLayer
from histopathology.models.components.kan.KANConv import KAN_Convolutional_Layer as KANCL

class FeatureHook:
    """
    Hook to capture intermediate activations for interpretability.
    """
    def __init__(self, name: str):
        self.name = name
        self.activations = None
        self.gradients = None

    def get_activation_hook(self):
        def hook(module, input, output):
            self.activations = output.detach()
        return hook

    def get_gradient_hook(self):
        def hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        return hook
        
    def clear(self):
        self.activations = None
        self.gradients = None


class LegacyAutoencoder_Encoder(nn.Module):
    """
    Encoder part of the autoencoder with KAN and ECA attention.
    Matches the architecture in the checkpoint.
    """
    def __init__(self, device: str = 'cpu', config=None):
        super(LegacyAutoencoder_Encoder, self).__init__()
        
        self.config = config or {"use_kan": True, "kan_options": {}}
        
        # Set up activation hooks for interpretability
        self.hooks = {}
        self.gradients = {}
        self.activations = {}
        
        # Input conv: 3 -> 384
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 384, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(384),
            nn.ELU(inplace=True),
        )

        # 384 -> 128
        self.encoder2 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
        )

        # 128 -> 64
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True)
        )

        # Attention mechanisms - use 5x5 kernel to match checkpoint
        if self.config.get("use_kan", True):
            self.kan = KANCL(
                n_convs=1,
                kernel_size=(5,5),  # Changed to 5x5 to match checkpoint
                padding=(2,2),      # Changed to 2,2 to match 5x5 kernel
                device=device
            )
        else:
            self.kan = nn.Identity()

        # Use the original ECA implementation with default gamma and b parameters
        self.ECA_Net = ECALayer(64) if self.config.get("use_eca", True) else nn.Identity()

        # 64 -> 128
        self.decoder1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 128, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True), 
        )

        # 128 -> 384
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 384, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(384),
            nn.ELU(inplace=True), 
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Main path
        out = self.encoder1(x)
        residual1 = out  # Store first residual (384 channels)
        
        out = self.encoder2(out)
        residual2 = out  # Store second residual (128 channels)
        
        out = self.encoder3(out)

        # Apply attention
        out = self.kan(out)
        out = self.ECA_Net(out)
        
        # Decode back
        out = self.decoder1(out) + residual2  # 128 channels
        out = self.decoder2(out) + residual1  # 384 channels
        
        # Store activations for interpretability
        self.activations = {
            'encoder1': self.encoder1[0].weight.detach(),
            'encoder2': self.encoder2[0].weight.detach(),
            'encoder3': self.encoder3[0].weight.detach(),
            'encoder_out': out.detach()
        }
        
        return out, residual1, residual2


class LegacyAutoencoder_BottleNeck(nn.Module):
    """
    Bottleneck module with BAM attention and VAE-style encoding.
    Modified to match checkpoint (using 128 channels in decoder).
    """
    def __init__(self, config=None): 
        super(LegacyAutoencoder_BottleNeck, self).__init__()
        self.config = config or {"use_bam": True}
        
        # Set up activation hooks for interpretability
        self.hooks = {}
        self.gradients = {}
        self.activations = {}

        self.encoder1 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(384),
            nn.ELU(inplace=True),
        )

        self.attn1 = BAM(384) if self.config.get("use_bam", True) else nn.Identity()

        self.encoder2 = nn.Sequential(
            nn.Conv2d(384, 16, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
        )

        self.attn2 = BAM(16) if self.config.get("use_bam", True) else nn.Identity()

        # Use 128 channels to match checkpoint
        self.decoder = nn.Sequential(
           nn.ConvTranspose2d(16, 128, kernel_size=3, padding=1, stride=2, output_padding=1),
           nn.BatchNorm2d(128),
           nn.ELU(inplace=True), 
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder1(x)
        x = self.attn1(x)
        x = self.encoder2(x)
        z = self.attn2(x)
        decoded = self.decoder(z)
        
        # Store activations for interpretability
        self.activations = {
            'encoder1': self.encoder1[0].weight.detach(),
            'attn1': z.detach() if not isinstance(self.attn1, nn.Identity) else None,
            'encoder2': self.encoder2[0].weight.detach(),
            'attn2': z.detach() if not isinstance(self.attn2, nn.Identity) else None,
            'bottleneck': z.detach(),
            'decoded': decoded.detach()
        }

        return decoded, z


class LegacyAutoencoder_Decoder(nn.Module):
    """
    Legacy decoder part of the autoencoder.
    Modified to match the architecture in the checkpoint exactly.
    """
    def __init__(self, device: str = 'cuda', config=None):
        super(LegacyAutoencoder_Decoder, self).__init__()
        
        self.config = config or {"use_kan": True, "kan_options": {}}
        
        # Set up activation hooks for interpretability
        self.hooks = {}
        self.gradients = {}
        self.activations = {}
        
        # Add channel adjustment layer to handle bottleneck output (128 channels) to encoder1 input (384 channels)
        self.channel_adj = nn.Conv2d(128, 384, kernel_size=1)  # 128 -> 384 channels
        
        # Corrected encoder1 dimensions to match the checkpoint exactly
        # The weight tensor shape should be [128, 384, 3, 3] where:
        # - 128 = number of output filters/channels
        # - 384 = number of input channels
        self.encoder1 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, padding=1, stride=2),  # in=384, out=128
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
        )
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2),  # Fixed: input channels changed from 384 to 128
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
        )
        
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
        )
        
        # Apply KAN with 5x5 kernel to match checkpoint
        if self.config.get("use_kan", True):
            self.kan = KANCL(
                n_convs=1,
                kernel_size=(5,5),  # Changed to 5x5 to match checkpoint
                padding=(2,2),      # Changed to 2,2 to match 5x5 kernel
                device=device
            )
        else:
            self.kan = nn.Identity()

        self.ECA_Net = ECALayer(64) if self.config.get("use_eca", True) else nn.Identity()
        
        # Update decoder dimensions
        self.decoder1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 128, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
        )
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 384, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(384),
            nn.ELU(inplace=True),
        )
        
        # Updated Output_Layer dimensions
        self.Output_Layer = nn.Sequential(
            nn.ConvTranspose2d(384, 3, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )
        
        # Update KAN reconstruction to use 3x3 kernel to match checkpoint
        if self.config.get("use_kan", True):
            self.reconstructtion = KANCL(  # Note the double 't' in the name
                n_convs=1,
                kernel_size=(3,3),  # Changed to 3x3 to match checkpoint
                padding=(1,1),      # Changed to 1,1 to match 3x3 kernel
                device=device
            )
        else:
            self.reconstructtion = nn.Identity()

    def forward(self, x: torch.Tensor, residualEnc1: torch.Tensor, residualEnc2: torch.Tensor) -> torch.Tensor:
        # Apply channel adjustment to match dimensions
        x = self.channel_adj(x)  # Convert from 128 to 384 channels
        
        # Forward through legacy architecture
        out = self.encoder1(x)
        out = self.encoder2(out)
        out = self.encoder3(out)
        
        # Apply KAN and ECA
        out = self.kan(out)
        out = self.ECA_Net(out)
        
        # Decoder path
        out = self.decoder1(out)
        out = self.decoder2(out)
        out = self.Output_Layer(out)
        
        # Apply final reconstruction KAN
        out = self.reconstructtion(out)
        
        return out


class LegacyDAE_KAN_Attention(nn.Module):
    """
    Legacy DAE-KAN-Attention model architecture that matches the checkpoint.
    Dimensions updated to match the checkpoint exactly:
    1. Bottleneck decoder uses 128 channels
    2. KAN layers use 5x5 kernel (25 dimensions) to match checkpoint
    3. Decoder architecture matches checkpoint exactly
    """
    def __init__(self, device: str = 'cuda', config=None):
        super().__init__()
        self.config = config or {
            "use_kan": True,
            "use_eca": True,
            "use_bam": True,
            "kan_options": {
                "kernel_size": [5,5],  # Main KAN layers use 5x5
                "padding": [2,2],      # Padding for 5x5 kernel
                "recon_kernel_size": [3,3],  # Reconstruction uses 3x3
                "recon_padding": [1,1]       # Padding for 3x3 kernel
            },
            "interpretability": {
                "enable_hooks": True,
                "enable_captum": True,
                "enable_gradcam": True,
                "store_activations": True
            }
        }
        
        self.ae_encoder = LegacyAutoencoder_Encoder(device=device, config=self.config)
        self.bottleneck = LegacyAutoencoder_BottleNeck(config=self.config)
        self.ae_decoder = LegacyAutoencoder_Decoder(device=device, config=self.config)
        
        # Set device for model
        self.device = device
        
        # Register hooks if interpretability is enabled
        if self.config.get("interpretability", {}).get("enable_hooks", True):
            self.register_hooks()
            
        # Save intermediate activations
        self.layer_activations = {}
        self.attention_maps = {}
        self.latent_representations = None

    def register_hooks(self):
        """Register hooks for all components"""
        pass  # Simplified for this example

    def forward(self, x):
        encoded, residual1, residual2 = self.ae_encoder(x)
        decoded, z = self.bottleneck(encoded)
        final_decoded = self.ae_decoder(decoded, residual1, residual2)
        
        # Store latent representations
        self.latent_representations = z
        
        return encoded, final_decoded, z

    def regularization_loss(self):
        """Calculate regularization loss for KAN layers"""
        reg_loss = 0.0
        if self.config.get("use_kan", True):
            # Add KAN regularization from encoder
            if hasattr(self.ae_encoder.kan, 'regularization_loss'):
                reg_loss += self.ae_encoder.kan.regularization_loss()
            
            # Add KAN regularization from decoder
            if hasattr(self.ae_decoder.kan, 'regularization_loss'):
                reg_loss += self.ae_decoder.kan.regularization_loss()
                
            # Note the different spelling in legacy model
            if hasattr(self.ae_decoder.reconstructtion, 'regularization_loss'):
                reg_loss += self.ae_decoder.reconstructtion.regularization_loss()
        
        return reg_loss
        
    def get_latent_representation(self):
        """Returns the latent space representation (bottleneck output)."""
        return self.latent_representations
