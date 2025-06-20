import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple, Union, Optional, Callable
import matplotlib.pyplot as plt
from torch.autograd import Variable
from histopathology.src.models.components.attention_mechanisms.bam import BAM
from histopathology.src.models.components.attention_mechanisms.eca import ECALayer
from histopathology.src.models.components.kan.kan_layer import KANLayer
from histopathology.src.models.components.kan.KANConv import KAN_Convolutional_Layer as KANCL

# Import Captum for attribution methods
try:
    from captum.attr import (
        IntegratedGradients,
        LayerGradCam,
        LayerAttribution,
        NeuronConductance,
        FeatureAblation,
        Occlusion
    )
    CAPTUM_AVAILABLE = True
except ImportError:
    print("Captum not available. Install with pip install captum for attribution methods.")
    CAPTUM_AVAILABLE = False

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


class Autoencoder_Encoder(nn.Module):
    """
    Encoder part of the autoencoder with KAN and ECA attention.
    Processes input images through encoding layers, applies attention,
    and partially decodes back with residual connections.
    """
    def __init__(self, device: str = 'cpu', config=None):
        super(Autoencoder_Encoder, self).__init__()
        
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

        # Attention mechanisms
        if self.config.get("use_kan", True):
            self.kan = KANCL(
                n_convs=1,
                kernel_size=tuple(self.config["kan_options"].get("kernel_size", [5,5])),
                padding=tuple(self.config["kan_options"].get("padding", [2,2])),
                device=device
            )
        else:
            self.kan = nn.Identity()

        self.ECA_Net = ECALayer(64) if self.config.get("use_eca", True) else nn.Identity()

        # 64 -> 128
        self.decoder1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 128, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True), 
        )

        # 128 -> 384 (fixed from 16 -> 128)
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 384, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(384),
            nn.ELU(inplace=True), 
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
        Returns:
            tuple containing:
            - output: Encoded and partially decoded features
            - residual1: First residual connection (384 channels)
            - residual2: Second residual connection (128 channels)
        """
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
        
    def register_hooks(self):
        """Register forward and backward hooks for capturing activations and gradients"""
        # Clear existing hooks
        for hook in self.hooks.values():
            hook.clear()
            
        # Create new hooks
        self.hooks['encoder1'] = FeatureHook('encoder1')
        self.hooks['encoder2'] = FeatureHook('encoder2')
        self.hooks['encoder3'] = FeatureHook('encoder3')
        self.hooks['kan'] = FeatureHook('kan')
        self.hooks['eca'] = FeatureHook('eca')
        
        # Register hooks
        self.encoder1[0].register_forward_hook(self.hooks['encoder1'].get_activation_hook())
        self.encoder2[0].register_forward_hook(self.hooks['encoder2'].get_activation_hook())
        self.encoder3[0].register_forward_hook(self.hooks['encoder3'].get_activation_hook())
        
        if not isinstance(self.kan, nn.Identity):
            self.kan.register_forward_hook(self.hooks['kan'].get_activation_hook())
        
        if not isinstance(self.ECA_Net, nn.Identity):
            self.ECA_Net.register_forward_hook(self.hooks['eca'].get_activation_hook())
            
    def get_attention_maps(self):
        """Return attention activation maps for interpretability"""
        attention_maps = {}
        
        if not isinstance(self.ECA_Net, nn.Identity) and self.hooks.get('eca') is not None:
            if self.hooks['eca'].activations is not None:
                attention_maps['eca'] = self.hooks['eca'].activations
                
        return attention_maps
        
    def get_layer_activations(self):
        """Return all layer activations for interpretability"""
        activations = {}
        
        for name, hook in self.hooks.items():
            if hook.activations is not None:
                activations[name] = hook.activations
                
        return activations


class Autoencoder_Decoder(nn.Module):
    """
    Decoder part of the autoencoder with KAN and ECA attention.
    Processes encoded features back to image space with residual connections.
    """
    def __init__(self, device: str = 'cuda', config=None):
        super(Autoencoder_Decoder, self).__init__()
        
        self.config = config or {"use_kan": True, "kan_options": {}}
        
        # Set up activation hooks for interpretability
        self.hooks = {}
        self.gradients = {}
        self.activations = {}
        
        # Channel adjustment for residualEnc2
        self.residual2_adj = nn.Conv2d(128, 384, kernel_size=1)
        
        # Apply KAN to the input features
        if self.config.get("use_kan", True):
            self.kan = KANCL(
                n_convs=1,
                kernel_size=tuple(self.config["kan_options"].get("kernel_size", [5,5])),
                padding=tuple(self.config["kan_options"].get("padding", [2,2])),
                device=device
            )
        else:
            self.kan = nn.Identity()

        # ECA attention on the features
        self.ECA_Net = ECALayer(384) if self.config.get("use_eca", True) else nn.Identity()

        # Upsampling layers
        self.up1 = nn.ConvTranspose2d(384, 384, kernel_size=4, stride=2, padding=1)  # 32x32 → 64x64
        self.up2 = nn.ConvTranspose2d(384, 384, kernel_size=4, stride=2, padding=1)  # 64x64 → 128x128
        
        # Residual upsampling
        self.residual_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Final convolution to get back to 3 channels (without activation)
        self.final_conv = nn.Sequential(
            nn.Conv2d(384, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1)
        )
        
        # Separate final activation
        self.final_activation = nn.Sigmoid()  # Ensure output is in [0,1] range for images
        
        # Final KAN layer for reconstruction refinement
        if self.config.get("use_kan", True):
            self.reconstruction = KANCL(
                n_convs=1,
                kernel_size=tuple(self.config["kan_options"].get("recon_kernel_size", [3,3])),
                padding=tuple(self.config["kan_options"].get("recon_padding", [1,1])),
                device=device
            )
        else:
            self.reconstruction = nn.Identity()

        # Enable debug mode for shape checking
        self.debug = True

    def forward(self, x: torch.Tensor, residualEnc1: torch.Tensor, residualEnc2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder
        Args:
            x: Input tensor from bottleneck (batch_size, 384, H/8, W/8)
            residualEnc1: First residual from encoder (batch_size, 384, H/2, W/2)
            residualEnc2: Second residual from encoder (batch_size, 128, H/4, W/4)
        Returns:
            Reconstructed image tensor (batch_size, 3, H, W)
        """
        # Print input shapes for debugging
        if self.debug:
            print(f"Decoder input shape: {x.shape}")
            print(f"ResidualEnc1 shape: {residualEnc1.shape}")
            print(f"ResidualEnc2 shape: {residualEnc2.shape}")

        # Apply KAN and ECA attention
        out = self.kan(x)
        out = self.ECA_Net(out)
        
        # First upsampling + residual2
        out = self.up1(out)  # [B, 384, 64, 64]
        if self.debug:
            print(f"After up1 shape: {out.shape}")
        
        # Adjust residualEnc2 channels and spatial dimensions
        residualEnc2 = self.residual2_adj(residualEnc2)  # [B, 384, 32, 32]
        residualEnc2 = self.residual_up(residualEnc2)    # [B, 384, 64, 64]
        out = out + residualEnc2
        
        # Second upsampling + residual1
        out = self.up2(out)  # [B, 384, 128, 128]
        if self.debug:
            print(f"After up2 shape: {out.shape}")
        
        # Adjust residualEnc1 spatial dimensions
        residualEnc1 = self.residual_up(residualEnc1)  # [B, 384, 128, 128]
        out = out + residualEnc1
        
        # Final convolution
        out = self.final_conv(out)
        
        # Apply final KAN layer for reconstruction refinement
        out = self.reconstruction(out)
        
        # Final activation to ensure [0,1] range
        out = self.final_activation(out)
        
        if self.debug:
            print(f"Final output shape: {out.shape}")
        
        # Store activations for interpretability
        self.activations = {
            # For KAN layer, store the output activations instead of weights
            'decoder_kan': self.hooks['kan'].activations.detach() if hasattr(self, 'hooks') and 'kan' in self.hooks and self.hooks['kan'].activations is not None else None,
            'up1': self.up1.weight.detach(),
            'up2': self.up2.weight.detach(),
            'final': out.detach()
        }
        
        return out
        
    def register_hooks(self):
        """Register forward and backward hooks for capturing activations and gradients"""
        # Clear existing hooks
        for hook in self.hooks.values():
            hook.clear()
            
        # Create new hooks
        self.hooks['kan'] = FeatureHook('kan')
        self.hooks['eca'] = FeatureHook('eca')
        self.hooks['up1'] = FeatureHook('up1')
        self.hooks['up2'] = FeatureHook('up2')
        self.hooks['final_conv'] = FeatureHook('final_conv')
        self.hooks['reconstruction'] = FeatureHook('reconstruction')
        
        # Register hooks
        if not isinstance(self.kan, nn.Identity):
            self.kan.register_forward_hook(self.hooks['kan'].get_activation_hook())
        
        if not isinstance(self.ECA_Net, nn.Identity):
            self.ECA_Net.register_forward_hook(self.hooks['eca'].get_activation_hook())
            
        self.up1.register_forward_hook(self.hooks['up1'].get_activation_hook())
        self.up2.register_forward_hook(self.hooks['up2'].get_activation_hook())
        self.final_conv[0].register_forward_hook(self.hooks['final_conv'].get_activation_hook())
        
        if not isinstance(self.reconstruction, nn.Identity):
            self.reconstruction.register_forward_hook(self.hooks['reconstruction'].get_activation_hook())
            
    def get_attention_maps(self):
        """Return attention activation maps for interpretability"""
        attention_maps = {}
        
        if not isinstance(self.ECA_Net, nn.Identity) and self.hooks.get('eca') is not None:
            if self.hooks['eca'].activations is not None:
                attention_maps['eca'] = self.hooks['eca'].activations
                
        return attention_maps
        
    def get_layer_activations(self):
        """Return all layer activations for interpretability"""
        activations = {}
        
        for name, hook in self.hooks.items():
            if hook.activations is not None:
                activations[name] = hook.activations
                
        return activations

class Autoencoder_BottleNeck(nn.Module):
    """
    Bottleneck module with BAM attention and VAE-style encoding
    """
    def __init__(self, config=None): 
        super(Autoencoder_BottleNeck, self).__init__()
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

        self.decoder = nn.Sequential(
           nn.ConvTranspose2d(16, 384, kernel_size=3, padding=1, stride=2, output_padding=1),  # Changed from 128 to 384
           nn.BatchNorm2d(384),
           nn.ELU(inplace=True), 
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through bottleneck
        Args:
            x: Input tensor (batch_size, 384, H/4, W/4)
        Returns:
            tuple containing:
            - decoded: Decoded features (batch_size, 384, H/4, W/4)
            - z: Bottleneck encoding (batch_size, 16, H/16, W/16)
        """
        x = self.encoder1(x)
        x = self.attn1(x)
        x = self.encoder2(x)
        z = self.attn2(x)
        decoded = self.decoder(z)  # Use z instead of x
        
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
        
    def register_hooks(self):
        """Register forward and backward hooks for capturing activations and gradients"""
        # Clear existing hooks
        for hook in self.hooks.values():
            hook.clear()
            
        # Create new hooks
        self.hooks['encoder1'] = FeatureHook('encoder1')
        self.hooks['attn1'] = FeatureHook('attn1')
        self.hooks['encoder2'] = FeatureHook('encoder2')
        self.hooks['attn2'] = FeatureHook('attn2')
        self.hooks['decoder'] = FeatureHook('decoder')
        
        # Register hooks
        self.encoder1[0].register_forward_hook(self.hooks['encoder1'].get_activation_hook())
        
        if not isinstance(self.attn1, nn.Identity):
            self.attn1.register_forward_hook(self.hooks['attn1'].get_activation_hook())
            
        self.encoder2[0].register_forward_hook(self.hooks['encoder2'].get_activation_hook())
        
        if not isinstance(self.attn2, nn.Identity):
            self.attn2.register_forward_hook(self.hooks['attn2'].get_activation_hook())
            
        self.decoder[0].register_forward_hook(self.hooks['decoder'].get_activation_hook())
        
    def get_attention_maps(self):
        """Return attention activation maps for interpretability"""
        attention_maps = {}
        
        if not isinstance(self.attn1, nn.Identity) and self.hooks.get('attn1') is not None:
            if self.hooks['attn1'].activations is not None:
                attention_maps['attn1'] = self.hooks['attn1'].activations
                
        if not isinstance(self.attn2, nn.Identity) and self.hooks.get('attn2') is not None:
            if self.hooks['attn2'].activations is not None:
                attention_maps['attn2'] = self.hooks['attn2'].activations
                
        return attention_maps
        
    def get_layer_activations(self):
        """Return all layer activations for interpretability"""
        activations = {}
        
        for name, hook in self.hooks.items():
            if hook.activations is not None:
                activations[name] = hook.activations
                
        return activations

class DAE_KAN_Attention(nn.Module):
    def __init__(self, device: str = 'cuda', config=None):
        super().__init__()
        self.config = config or {
            "use_kan": True,
            "use_eca": True,
            "use_bam": True,
            "kan_options": {
                "kernel_size": [5,5],
                "padding": [2,2],
                "recon_kernel_size": [3,3],
                "recon_padding": [1,1]
            },
            "interpretability": {
                "enable_hooks": True,
                "enable_captum": True,
                "enable_gradcam": True,
                "store_activations": True
            }
        }
        self.ae_encoder = Autoencoder_Encoder(device=device, config=self.config)
        self.bottleneck = Autoencoder_BottleNeck(config=self.config)
        self.ae_decoder = Autoencoder_Decoder(device=device, config=self.config)
        
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
        self.ae_encoder.register_hooks()
        self.bottleneck.register_hooks()
        self.ae_decoder.register_hooks()

    def forward(self, x):
        encoded, residual1, residual2 = self.ae_encoder(x)
        decoded, z = self.bottleneck(encoded)
        final_decoded = self.ae_decoder(decoded, residual1, residual2)
        
        # Store activations for interpretability if enabled
        if self.config.get("interpretability", {}).get("store_activations", True):
            self.layer_activations = {
                'encoder': self.ae_encoder.get_layer_activations(),
                'bottleneck': self.bottleneck.get_layer_activations(),
                'decoder': self.ae_decoder.get_layer_activations()
            }
            
            self.attention_maps = {
                'encoder': self.ae_encoder.get_attention_maps(),
                'bottleneck': self.bottleneck.get_attention_maps(),
                'decoder': self.ae_decoder.get_attention_maps()
            }
            
            self.latent_representations = z.detach()

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
            if hasattr(self.ae_decoder.reconstruction, 'regularization_loss'):
                reg_loss += self.ae_decoder.reconstruction.regularization_loss()
        
        return reg_loss
        
    def get_layer_activations(self):
        """
        Returns all layer activations for interpretability and visualization.
        
        Returns:
            dict: Dictionary containing all stored layer activations
        """
        return self.layer_activations
        
    def get_attention_maps(self):
        """
        Returns attention maps from all attention modules.
        
        Returns:
            dict: Dictionary containing all stored attention maps
        """
        return self.attention_maps
        
    def get_latent_representation(self):
        """
        Returns the latent space representation (bottleneck output).
        
        Returns:
            torch.Tensor: Latent space representation
        """
        return self.latent_representations
        
    def visualize_layer_activations(self, layer_name, sample_idx=0, channel_idx=None, figsize=(12, 8)):
        """
        Visualize activations of a specific layer.
        
        Args:
            layer_name (str): Name of the layer in format 'component/layer' (e.g. 'encoder/encoder1')
            sample_idx (int): Index of the sample in batch to visualize
            channel_idx (int, optional): Channel to visualize (None for all channels)
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Figure with activation visualizations
        """
        component, layer = layer_name.split('/')
        
        if component not in self.layer_activations or layer not in self.layer_activations[component]:
            raise ValueError(f"Layer {layer_name} not found in activations")
            
        activations = self.layer_activations[component][layer]
        
        if activations is None:
            raise ValueError(f"No activations stored for layer {layer_name}")
        
        # Get single sample activations
        sample_activations = activations[sample_idx].cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(figsize=figsize)
        
        if channel_idx is not None:
            # Show single channel
            im = axes.imshow(sample_activations[channel_idx], cmap='viridis')
            axes.set_title(f"{layer_name} (Channel {channel_idx})")
            plt.colorbar(im, ax=axes)
        else:
            # Show mean across channels
            mean_activation = np.mean(sample_activations, axis=0)
            im = axes.imshow(mean_activation, cmap='viridis')
            axes.set_title(f"{layer_name} (Mean Activation)")
            plt.colorbar(im, ax=axes)
            
        return fig
        
    def visualize_attention_maps(self, attention_name, sample_idx=0, figsize=(12, 8)):
        """
        Visualize attention maps.
        
        Args:
            attention_name (str): Name of the attention in format 'component/attention' (e.g. 'encoder/eca')
            sample_idx (int): Index of the sample in batch to visualize
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Figure with attention visualizations
        """
        component, attention = attention_name.split('/')
        
        if component not in self.attention_maps or attention not in self.attention_maps[component]:
            raise ValueError(f"Attention {attention_name} not found in attention maps")
            
        attention_map = self.attention_maps[component][attention]
        
        if attention_map is None:
            raise ValueError(f"No attention map stored for {attention_name}")
        
        # Get single sample attention map
        sample_attention = attention_map[sample_idx].cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(figsize=figsize)
        
        # Show mean across channels for attention map
        mean_attention = np.mean(sample_attention, axis=0)
        im = axes.imshow(mean_attention, cmap='hot')
        axes.set_title(f"{attention_name} Attention Map")
        plt.colorbar(im, ax=axes)
            
        return fig
        
    def visualize_latent_space(self, method='pca', figsize=(12, 8)):
        """
        Visualize the latent space representation.
        
        Args:
            method (str): Dimensionality reduction method ('pca', 'tsne', or 'umap')
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Figure with latent space visualization
        """
        if self.latent_representations is None:
            raise ValueError("No latent representations stored")
            
        # Flatten latent representation for visualization
        latent_flat = self.latent_representations.view(self.latent_representations.size(0), -1).cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(figsize=figsize)
        
        if method == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            latent_2d = pca.fit_transform(latent_flat)
            axes.scatter(latent_2d[:, 0], latent_2d[:, 1])
            axes.set_title("PCA of Latent Space")
            
        elif method == 'tsne':
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2)
            latent_2d = tsne.fit_transform(latent_flat)
            axes.scatter(latent_2d[:, 0], latent_2d[:, 1])
            axes.set_title("t-SNE of Latent Space")
            
        elif method == 'umap':
            try:
                import umap
                reducer = umap.UMAP()
                latent_2d = reducer.fit_transform(latent_flat)
                axes.scatter(latent_2d[:, 0], latent_2d[:, 1])
                axes.set_title("UMAP of Latent Space")
            except ImportError:
                axes.text(0.5, 0.5, "UMAP not installed. Install with pip install umap-learn", 
                         ha='center', va='center')
        
        return fig
        
    def cluster_latent_space(self, n_clusters=5, method='kmeans'):
        """
        Perform clustering on the latent space.
        
        Args:
            n_clusters (int): Number of clusters
            method (str): Clustering method ('kmeans' or 'hierarchical')
            
        Returns:
            tuple: (cluster_labels, cluster_centers, silhouette_score)
        """
        if self.latent_representations is None:
            raise ValueError("No latent representations stored")
            
        # Flatten latent representation for clustering
        latent_flat = self.latent_representations.view(self.latent_representations.size(0), -1).cpu().numpy()
        
        if method == 'kmeans':
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(latent_flat)
            centers = kmeans.cluster_centers_
            
            if len(np.unique(labels)) > 1:  # Need at least 2 clusters for silhouette score
                silhouette = silhouette_score(latent_flat, labels)
            else:
                silhouette = 0
                
            return labels, centers, silhouette
            
        elif method == 'hierarchical':
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.metrics import silhouette_score
            
            hc = AgglomerativeClustering(n_clusters=n_clusters)
            labels = hc.fit_predict(latent_flat)
            
            # Hierarchical clustering doesn't provide centers directly
            # Calculate centers as mean of points in each cluster
            centers = np.array([latent_flat[labels == i].mean(axis=0) for i in range(n_clusters)])
            
            if len(np.unique(labels)) > 1:  # Need at least 2 clusters for silhouette score
                silhouette = silhouette_score(latent_flat, labels)
            else:
                silhouette = 0
                
            return labels, centers, silhouette
            
        else:
            raise ValueError(f"Unknown clustering method: {method}")
            
    def get_feature_importance(self, input_tensor, target_layer, method='integrated_gradients'):
        """
        Compute feature importance using Captum attribution methods.
        
        Args:
            input_tensor (torch.Tensor): Input to analyze
            target_layer (str): Target layer name ('encoder', 'bottleneck', 'decoder')
            method (str): Attribution method ('integrated_gradients', 'gradcam', 'conductance')
            
        Returns:
            torch.Tensor: Attribution scores
        """
        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum is not available. Install with pip install captum.")
            
        # Create a model wrapper for Captum
        class ModelWrapper(nn.Module):
            def __init__(self, model, target='decoded'):
                super().__init__()
                self.model = model
                self.target = target
                
            def forward(self, x):
                encoded, decoded, z = self.model(x)
                if self.target == 'encoded':
                    return encoded
                elif self.target == 'decoded':
                    return decoded
                elif self.target == 'latent':
                    return z
                    
        # Use decoded output as default target
        wrapper = ModelWrapper(self, target='decoded')
        
        if method == 'integrated_gradients':
            ig = IntegratedGradients(wrapper)
            attributions = ig.attribute(input_tensor, n_steps=50)
            return attributions
            
        elif method == 'gradcam':
            # Get target layer for GradCAM
            if target_layer == 'encoder':
                layer = self.ae_encoder.encoder3[0]  # Last conv layer in encoder
            elif target_layer == 'bottleneck':
                layer = self.bottleneck.encoder2[0]  # Last conv layer in bottleneck
            elif target_layer == 'decoder':
                layer = self.ae_decoder.final_conv[0]  # Last conv layer in decoder
                
            gradcam = LayerGradCam(wrapper, layer)
            attributions = gradcam.attribute(input_tensor)
            return LayerAttribution.interpolate(attributions, input_tensor.shape[2:])
            
        elif method == 'conductance':
            # Get target layer for Neuron Conductance
            if target_layer == 'encoder':
                layer = self.ae_encoder.encoder3[0]  # Last conv layer in encoder
            elif target_layer == 'bottleneck':
                layer = self.bottleneck.encoder2[0]  # Last conv layer in bottleneck
            elif target_layer == 'decoder':
                layer = self.ae_decoder.final_conv[0]  # Last conv layer in decoder
                
            # Use first neuron for demonstration
            conductance = NeuronConductance(wrapper, layer)
            attributions = conductance.attribute(input_tensor, neuron_index=0)
            return attributions
            
        else:
            raise ValueError(f"Unknown attribution method: {method}")
            
    def compute_gradcam(self, input_tensor, target_layer_name):
        """
        Compute GradCAM activation maps manually.
        
        Args:
            input_tensor (torch.Tensor): Input to analyze
            target_layer_name (str): Name of target layer in format 'component/layer'
            
        Returns:
            numpy.ndarray: GradCAM activation maps
        """
        component, layer = target_layer_name.split('/')
        
        # Ensure model is in eval mode
        self.eval()
        
        # Forward pass with gradient calculation
        input_tensor.requires_grad_()
        encoded, decoded, z = self(input_tensor)
        
        # Use decoded image as target for GradCAM
        target = decoded
        
        # Get target layer activations
        if component == 'encoder':
            if layer == 'encoder1':
                target_activations = self.ae_encoder.hooks['encoder1'].activations
            elif layer == 'encoder2':
                target_activations = self.ae_encoder.hooks['encoder2'].activations
            elif layer == 'encoder3':
                target_activations = self.ae_encoder.hooks['encoder3'].activations
            elif layer == 'kan':
                target_activations = self.ae_encoder.hooks['kan'].activations
            elif layer == 'eca':
                target_activations = self.ae_encoder.hooks['eca'].activations
        elif component == 'bottleneck':
            if layer == 'encoder1':
                target_activations = self.bottleneck.hooks['encoder1'].activations
            elif layer == 'attn1':
                target_activations = self.bottleneck.hooks['attn1'].activations
            elif layer == 'encoder2':
                target_activations = self.bottleneck.hooks['encoder2'].activations
            elif layer == 'attn2':
                target_activations = self.bottleneck.hooks['attn2'].activations
        elif component == 'decoder':
            if layer == 'kan':
                target_activations = self.ae_decoder.hooks['kan'].activations
            elif layer == 'eca':
                target_activations = self.ae_decoder.hooks['eca'].activations
            elif layer == 'up1':
                target_activations = self.ae_decoder.hooks['up1'].activations
            elif layer == 'up2':
                target_activations = self.ae_decoder.hooks['up2'].activations
            elif layer == 'final_conv':
                target_activations = self.ae_decoder.hooks['final_conv'].activations
            elif layer == 'reconstruction':
                target_activations = self.ae_decoder.hooks['reconstruction'].activations
                
        if target_activations is None:
            raise ValueError(f"No activations found for layer {target_layer_name}")
            
        # Calculate gradients with respect to activations
        target.sum().backward(retain_graph=True)
        
        # Get gradients
        gradients = None
        if component == 'encoder':
            if layer in self.ae_encoder.hooks and hasattr(self.ae_encoder.hooks[layer], 'gradients'):
                gradients = self.ae_encoder.hooks[layer].gradients
        elif component == 'bottleneck':
            if layer in self.bottleneck.hooks and hasattr(self.bottleneck.hooks[layer], 'gradients'):
                gradients = self.bottleneck.hooks[layer].gradients
        elif component == 'decoder':
            if layer in self.ae_decoder.hooks and hasattr(self.ae_decoder.hooks[layer], 'gradients'):
                gradients = self.ae_decoder.hooks[layer].gradients
                
        if gradients is None:
            # If gradients aren't captured by hooks (e.g. if module doesn't support it),
            # we can use gradients from parameters as a fallback
            print(f"Warning: No gradients found for layer {target_layer_name}, using parameter gradients instead")
            
            # Find the target module and get its parameters' gradients
            if component == 'encoder':
                if layer == 'encoder1':
                    gradients = self.ae_encoder.encoder1[0].weight.grad
                elif layer == 'encoder2':
                    gradients = self.ae_encoder.encoder2[0].weight.grad
                elif layer == 'encoder3':
                    gradients = self.ae_encoder.encoder3[0].weight.grad
            elif component == 'bottleneck':
                if layer == 'encoder1':
                    gradients = self.bottleneck.encoder1[0].weight.grad
                elif layer == 'encoder2':
                    gradients = self.bottleneck.encoder2[0].weight.grad
            elif component == 'decoder':
                if layer == 'up1':
                    gradients = self.ae_decoder.up1.weight.grad
                elif layer == 'up2':
                    gradients = self.ae_decoder.up2.weight.grad
                elif layer == 'final_conv':
                    gradients = self.ae_decoder.final_conv[0].weight.grad
        
        # Compute GradCAM
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        # Weight activation maps by importance (gradients)
        for i in range(target_activations.size(1)):
            target_activations[:, i, :, :] *= pooled_gradients[i]
            
        # Average across channels
        heatmap = torch.mean(target_activations, dim=1).detach()
        
        # ReLU on heatmap
        heatmap = torch.maximum(heatmap, torch.zeros_like(heatmap))
        
        # Normalize heatmap
        heatmap_min = heatmap.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        heatmap_max = heatmap.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min + 1e-8)
        
        return heatmap.cpu().numpy()
        
    def visualize_gradcam(self, input_tensor, target_layer_name, sample_idx=0, figsize=(12, 8)):
        """
        Visualize GradCAM activation maps.
        
        Args:
            input_tensor (torch.Tensor): Input to analyze
            target_layer_name (str): Name of target layer in format 'component/layer'
            sample_idx (int): Index of the sample in batch to visualize
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Figure with GradCAM visualization
        """
        # Compute GradCAM
        heatmaps = self.compute_gradcam(input_tensor, target_layer_name)
        
        # Get sample heatmap
        heatmap = heatmaps[sample_idx]
        
        # Get original image
        original_img = input_tensor[sample_idx].cpu().detach().permute(1, 2, 0).numpy()
        
        # Normalize image for display if needed
        if original_img.max() > 1.0:
            original_img = original_img / 255.0
            
        # Resize heatmap to match image size if needed
        if heatmap.shape != original_img.shape[:2]:
            from skimage.transform import resize
            heatmap = resize(heatmap, original_img.shape[:2], preserve_range=True)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot original image
        axes[0].imshow(original_img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Plot heatmap
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title(f"GradCAM: {target_layer_name}")
        axes[1].axis('off')
        
        # Plot overlay
        heatmap_rgb = plt.cm.jet(heatmap)[:, :, :3]
        overlay = original_img * 0.7 + heatmap_rgb * 0.3
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis('off')
        
        plt.tight_layout()
        return fig

class KAN_feature_extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.flat = nn.Flatten()
        self.KAN1 = KANLayer(16**3, 4096)
        self.KAN2 = KANLayer(4096, 2048)

    def forward(self, x):
        x = self.flat(x)
        x = self.KAN1(x)
        x = self.KAN2(x)

        return x
        

if __name__ == '__main__':
    torch.cuda.empty_cache()
    from torchsummary import summary
    x_d = torch.randn(2, 128, 128, 128)
    x_e = torch.randn(10, 3, 128, 128).to('cuda')
    x_b = torch.randn(2, 384, 256, 256)
    model_en = Autoencoder_Encoder(device='cuda').to('cuda')
    model_de = Autoencoder_Decoder(device='cuda').to('cuda')
    model_bn = Autoencoder_BottleNeck().to('cuda')
    complete = DAE_KAN_Attention().to('cuda')
    x, y, z = complete(x_e)

    summary(complete, x_e)
    print(x.shape)
    # flat = nn.Flatten().to('cuda')
    # flatten = flat(z)
    #print(flatten.shape)
    #kan_feat = KANLayer(4096, 2048, device='cuda')

    #print(kan_feat(z).shape)
    #fe = KAN_feature_extractor().to('cuda')
    #summary(fe, z)
    #print(z)
    #print(y[1].shape)


    #out = model_de(x)
    #print(z)
    #summary(model_bn, x_b)

