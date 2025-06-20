import os
import torch
from torch import nn
import numpy as np
from functools import partial
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
import torchmetrics
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
from torch.utils.data import DataLoader
from histopathology.src.models.autoencoders.dae_kan_attention.model import DAE_KAN_Attention
from histopathology.src.data_pipeline.histopathology_dataset import ImageDataset, create_dataset
import wandb
import torchvision.transforms as T
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import psutil
import GPUtil
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import yaml
from pathlib import Path
import io
from PIL import Image

# Try to import umap for dimensionality reduction
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with pip install umap-learn for enhanced dimensionality reduction")

# Try to import captum for attribution methods
try:
    from captum.attr import IntegratedGradients, LayerGradCam, LayerAttribution, Occlusion
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    print("Captum not available. Install with pip install captum for attribution visualizations")
# Helper functions for visualization and logging
def fig_to_wandb_image(fig, close_fig=True):
    """Convert matplotlib figure to wandb Image"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    if close_fig:
        plt.close(fig)
    return wandb.Image(img)

def get_system_metrics():
    """Get system metrics for GPU and CPU usage"""
    metrics = {}
    
    # CPU metrics
    metrics['cpu_percent'] = psutil.cpu_percent()
    metrics['ram_percent'] = psutil.virtual_memory().percent
    
    # GPU metrics if available
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            metrics['gpu_utilization'] = gpus[0].load * 100
            metrics['gpu_memory_percent'] = gpus[0].memoryUtil * 100
            metrics['gpu_temperature'] = gpus[0].temperature
    except Exception as e:
        print(f"Error getting GPU metrics: {e}")
    
    return metrics

class AdvancedWandbCallback(pl.Callback):
    """Custom callback for advanced WandB logging including system metrics and visualizations"""
    def __init__(
        self, 
        log_system_metrics: bool = True,
        log_system_freq: int = 100,
        log_gradients: bool = True,
        log_gradients_freq: int = 100,
        log_latent_freq: int = 500,
        log_attention_freq: int = 1000,
        log_gradcam_freq: int = 2000,
        track_params: bool = True,
        track_cluster_evolution: bool = True,
        n_clusters: int = 5,
    ):
        super().__init__()
        self.log_system_metrics = log_system_metrics
        self.log_system_freq = log_system_freq
        self.log_gradients = log_gradients
        self.log_gradients_freq = log_gradients_freq
        self.log_latent_freq = log_latent_freq
        self.log_attention_freq = log_attention_freq
        self.log_gradcam_freq = log_gradcam_freq
        self.track_params = track_params
        self.track_cluster_evolution = track_cluster_evolution
        self.n_clusters = n_clusters
        self.last_system_metrics_time = time.time()
        self.last_latent_vis_time = time.time()
        self.last_attention_vis_time = time.time()
        self.last_gradcam_vis_time = time.time()
        self.cluster_history = []
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Log system metrics with throttling to reduce overhead
        if self.log_system_metrics and batch_idx % self.log_system_freq == 0:
            metrics = get_system_metrics()
            trainer.logger.experiment.log(
                {f"system/{k}": v for k, v in metrics.items()},
                commit=False
            )
            
        # Log gradients periodically
        if self.log_gradients and batch_idx % self.log_gradients_freq == 0:
            for name, param in pl_module.named_parameters():
                if param.requires_grad and param.grad is not None:
                    trainer.logger.experiment.log({
                        f"gradients/{name}_norm": torch.norm(param.grad).item(),
                        f"weights/{name}_norm": torch.norm(param).item()
                    }, commit=False)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Only run these analyses on main validation at end of epoch
        if trainer.sanity_checking:
            return
            
        # Get validation batch for analysis
        val_batch = next(iter(trainer.val_dataloaders[0]))
        val_batch = val_batch.to(pl_module.device)
        
        # Log latent space visualizations
        self._log_latent_space(trainer, pl_module, val_batch)
        
        # Log attention maps
        self._log_attention_maps(trainer, pl_module, val_batch)
        
        # Log GradCAM visualizations
        if CAPTUM_AVAILABLE:
            self._log_gradcam(trainer, pl_module, val_batch)
        
    def _log_latent_space(self, trainer, pl_module, batch):
        """Log latent space visualizations and clustering"""
        with torch.no_grad():
            _, _, latent = pl_module.model(batch[:32])  # Limit to 32 samples for efficiency
        
        # Flatten latent representation
        latent_flat = latent.view(latent.size(0), -1).cpu().numpy()
        
        # Run dimensionality reduction
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(latent_flat)
        
        tsne = TSNE(n_components=2, perplexity=min(5, len(latent_flat)-1))
        tsne_result = tsne.fit_transform(latent_flat)
        
        # Create visualization figures
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(pca_result[:, 0], pca_result[:, 1])
        axes[0].set_title('PCA of Latent Space')
        
        axes[1].scatter(tsne_result[:, 0], tsne_result[:, 1])
        axes[1].set_title('t-SNE of Latent Space')
        
        plt.tight_layout()
        trainer.logger.experiment.log({"latent/projections": fig_to_wandb_image(fig)}, commit=False)
        
        # Run clustering analysis
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        clusters = kmeans.fit_predict(latent_flat)
        
        # Calculate silhouette score if we have enough samples and clusters
        if len(np.unique(clusters)) > 1 and len(latent_flat) > self.n_clusters:
            sil_score = silhouette_score(latent_flat, clusters)
            trainer.logger.experiment.log({"clustering/silhouette_score": sil_score}, commit=False)
        
        # Plot clusters
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # PCA with clusters
        for i in range(self.n_clusters):
            mask = clusters == i
            axes[0].scatter(pca_result[mask, 0], pca_result[mask, 1], label=f'Cluster {i}')
        axes[0].set_title('PCA with Clusters')
        axes[0].legend()
        
        # t-SNE with clusters
        for i in range(self.n_clusters):
            mask = clusters == i
            axes[1].scatter(tsne_result[mask, 0], tsne_result[mask, 1], label=f'Cluster {i}')
        axes[1].set_title('t-SNE with Clusters')
        axes[1].legend()
        
        plt.tight_layout()
        trainer.logger.experiment.log({"clustering/visualization": fig_to_wandb_image(fig)}, commit=False)
        
        # Track cluster evolution if enabled
        if self.track_cluster_evolution:
            self.cluster_history.append({
                'epoch': trainer.current_epoch,
                'centers': kmeans.cluster_centers_,
                'labels': clusters,
            })
            
            # If we have at least two epochs of history, plot cluster movement
            if len(self.cluster_history) > 1:
                prev_centers = self.cluster_history[-2]['centers']
                curr_centers = self.cluster_history[-1]['centers']
                
                # Project the cluster centers
                all_centers = np.vstack([prev_centers, curr_centers])
                pca_centers = pca.transform(all_centers)
                
                prev_pca = pca_centers[:len(prev_centers)]
                curr_pca = pca_centers[len(prev_centers):]
                
                # Plot cluster movement
                fig, ax = plt.subplots(figsize=(8, 8))
                for i in range(self.n_clusters):
                    ax.scatter(prev_pca[i, 0], prev_pca[i, 1], marker='o', s=100, label=f'Prev {i}')
                    ax.scatter(curr_pca[i, 0], curr_pca[i, 1], marker='x', s=100, label=f'Curr {i}')
                    ax.arrow(prev_pca[i, 0], prev_pca[i, 1], 
                             curr_pca[i, 0] - prev_pca[i, 0], 
                             curr_pca[i, 1] - prev_pca[i, 1], 
                             width=0.01, head_width=0.05, length_includes_head=True)
                
                ax.set_title(f'Cluster Movement (Epochs {trainer.current_epoch-1} to {trainer.current_epoch})')
                plt.tight_layout()
                trainer.logger.experiment.log({"clustering/movement": fig_to_wandb_image(fig)}, commit=False)
    
    def _log_attention_maps(self, trainer, pl_module, batch):
        """Log attention maps from the model"""
        with torch.no_grad():
            # Forward pass to capture activations
            pl_module.model(batch[:4])  # Only process 4 samples for efficiency
            
            # Get attention maps
            attention_maps = pl_module.model.get_attention_maps()
            
            # Create visualizations for each attention mechanism
            for component, maps in attention_maps.items():
                for attn_name, attn_map in maps.items():
                    # Skip if no attention map available
                    if attn_map is None:
                        continue
                        
                    # Create a grid of attention maps for the samples
                    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
                    axes = axes.flatten()
                    
                    for i in range(min(4, len(attn_map))):
                        # Get sample attention and average across channels
                        sample_attn = attn_map[i].cpu().numpy()
                        mean_attn = np.mean(sample_attn, axis=0)
                        
                        # Plot
                        im = axes[i].imshow(mean_attn, cmap='hot')
                        axes[i].set_title(f"Sample {i}")
                        axes[i].axis('off')
                    
                    plt.colorbar(im, ax=axes.tolist())
                    plt.tight_layout()
                    
                    # Log to wandb
                    trainer.logger.experiment.log({
                        f"attention/{component}_{attn_name}": fig_to_wandb_image(fig)
                    }, commit=False)
    
    def _log_gradcam(self, trainer, pl_module, batch):
        """Log GradCAM and attribution visualizations"""
        if not CAPTUM_AVAILABLE:
            return
            
        # Create visualizations for a subset of samples
        sample_batch = batch[:2]  # Only 2 samples for efficiency
        
        # List of target layers to visualize
        target_layers = [
            'encoder/encoder3',
            'bottleneck/encoder2',
            'decoder/final_conv'
        ]
        
        for layer_name in target_layers:
            # Compute GradCAM using the model's built-in method
            fig = pl_module.model.visualize_gradcam(
                sample_batch, 
                target_layer_name=layer_name,
                sample_idx=0,
                figsize=(12, 4)
            )
            
            # Log to wandb
            trainer.logger.experiment.log({
                f"gradcam/{layer_name}": fig_to_wandb_image(fig)
            }, commit=False)
            
        # Log integrated gradients attribution if available
        if hasattr(pl_module.model, 'get_feature_importance'):
            try:
                attributions = pl_module.model.get_feature_importance(
                    sample_batch, 
                    target_layer='decoder',
                    method='integrated_gradients'
                )
                
                # Visualize attributions
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original image
                axes[0].imshow(sample_batch[0].cpu().permute(1, 2, 0))
                axes[0].set_title("Original")
                axes[0].axis('off')
                
                # Attribution map (channel-wise mean)
                attr_mean = attributions[0].abs().mean(dim=0).cpu()
                im = axes[1].imshow(attr_mean, cmap='viridis')
                axes[1].set_title("Attribution Magnitude")
                axes[1].axis('off')
                plt.colorbar(im, ax=axes[1])
                
                # Overlay
                attr_norm = attr_mean / attr_mean.max()
                overlay = sample_batch[0].cpu().permute(1, 2, 0).numpy() * 0.7
                overlay = overlay + plt.cm.viridis(attr_norm)[:, :, :3] * 0.3
                axes[2].imshow(overlay)
                axes[2].set_title("Overlay")
                axes[2].axis('off')
                
                plt.tight_layout()
                trainer.logger.experiment.log({
                    "attribution/integrated_gradients": fig_to_wandb_image(fig)
                }, commit=False)
            except Exception as e:
                print(f"Error generating attributions: {e}")

class RobustDAE(pl.LightningModule):
    def __init__(
        self, 
        mask_ratio=0.3, 
        noise_level=0.2,
        config=None,
        wandb_config=None
    ):
        super().__init__()
        
        # Initialize configuration
        self.config = config or {}
        self.wandb_config = wandb_config or {}
        
        # Initialize model with config
        model_config = self.config.get('model', {})
        self.model = DAE_KAN_Attention(config=model_config)
        
        # Training parameters
        self.mask_ratio = mask_ratio
        self.noise_level = noise_level
        self.batch_visualization_indices = list(range(4))  # Indices for visualization
        
        # Save hyperparameters for tracking
        self.save_hyperparameters(ignore=['config', 'wandb_config'])
        
        # Initialize metrics
        self.train_psnr = PeakSignalNoiseRatio()
        self.train_ssim = StructuralSimilarityIndexMeasure()
        self.val_psnr = PeakSignalNoiseRatio()
        self.val_ssim = StructuralSimilarityIndexMeasure()
        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.val_lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')

        # Curriculum learning parameters
        self.register_buffer('curriculum_step', torch.tensor(0))
        self.max_curriculum_steps = 1000
        
        # Augmentations
        self.augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
        ])

    def apply_masking(self, x):
        """Apply random rectangular masking with curriculum learning"""
        b, c, h, w = x.shape
        
        # Curriculum-adjusted mask parameters
        max_mask_size = min(h, w) * (0.3 + 0.2 * (self.curriculum_step/self.max_curriculum_steps))
        min_mask_size = min(h, w) * 0.1
        
        masked_x = x.clone()
        for i in range(b):
            # Random mask position and size
            mask_h = int(torch.randint(int(min_mask_size), int(max_mask_size), (1,)))
            mask_w = int(torch.randint(int(min_mask_size), int(max_mask_size), (1,)))
            pos_h = int(torch.randint(0, h - mask_h, (1,)))
            pos_w = int(torch.randint(0, w - mask_w, (1,)))
            
            # Apply mask
            masked_x[i, :, pos_h:pos_h+mask_h, pos_w:pos_w+mask_w] = 0
            
        return masked_x
        
    def log_gradient_histograms(self):
        """Log histograms of model gradients"""
        # Skip if WandB is disabled
        wandb_enabled = os.environ.get("WANDB_MODE", "").lower() != "disabled" and os.environ.get("WANDB_MODE", "").lower() != "dryrun"
        if not wandb_enabled:
            return
            
        try:
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.logger.experiment.log({
                        f"grad_hist/{name}": wandb.Histogram(param.grad.detach().cpu().numpy())
                    }, commit=False)
        except Exception as e:
            print(f"Error in log_gradient_histograms: {e}")

    def on_train_start(self):
        """Log model architecture and sample batch images at start of training"""
        # Skip if WandB is disabled
        wandb_enabled = os.environ.get("WANDB_MODE", "").lower() != "disabled" and os.environ.get("WANDB_MODE", "").lower() != "dryrun"
        if not wandb_enabled:
            return
            
        try:
            # Log model summary as text
            model_summary = str(self.model)
            self.logger.experiment.log({
                "model/architecture": model_summary,
                "model/total_parameters": sum(p.numel() for p in self.model.parameters()),
                "model/trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            })
            
            # Create model graph if a sample batch is available
            if hasattr(self, "sample_batch") and self.sample_batch is not None:
                self.logger.watch(self.model, log="all")
        except Exception as e:
            print(f"Error in on_train_start: {e}")
    
    def training_step(self, batch, batch_idx):
        # Store sample batch for visualization if needed
        if not hasattr(self, "sample_batch"):
            self.sample_batch = batch.clone()
            
        x = self.augment(batch)
        
        # Apply curriculum-based masking and noise
        masked_x = self.apply_masking(x)
        noisy_x = masked_x + torch.randn_like(masked_x) * self.noise_level
        
        # Forward pass
        encoded, decoded, z = self.model(noisy_x)

        # Reconstruction loss
        mse_loss = nn.MSELoss()(decoded, x)
        reg_loss = self.model.regularization_loss()
        
        # Curriculum-aware loss weighting
        curriculum_weight = 0.5 * (1 + torch.cos(torch.pi * self.curriculum_step/self.max_curriculum_steps))
        loss = mse_loss + reg_loss * curriculum_weight
        
        # Update curriculum step
        self.curriculum_step += 1

        # Update metrics
        self.train_psnr(decoded, x)
        self.train_ssim(decoded, x)
        
        # Basic metric logging
        metrics_dict = {
            'train_loss': loss,
            'train_mse_loss': mse_loss,
            'train_reg_loss': reg_loss,
            'train_psnr': self.train_psnr,
            'train_ssim': self.train_ssim,
            'curriculum_weight': curriculum_weight,
            'learning_rate': self.trainer.optimizers[0].param_groups[0]['lr']
        }
        
        # Check if WandB is enabled
        wandb_enabled = os.environ.get("WANDB_MODE", "").lower() != "disabled" and os.environ.get("WANDB_MODE", "").lower() != "dryrun"
        
        # Add WandB-specific logging only if enabled
        if wandb_enabled and batch_idx % 100 == 0:
            try:
                # Add histogram of latent space values
                metrics_dict['latent/histogram'] = wandb.Histogram(z.detach().cpu().flatten().numpy())
                
                # Log gradient histograms periodically (only on certain steps to reduce overhead)
                if self.global_step % 500 == 0:
                    self.log_gradient_histograms()
            except Exception as e:
                print(f"Error in WandB logging during training step: {e}")
                
        # Log all metrics
        self.log_dict(metrics_dict, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        encoded, decoded, z = self.model(x)

        mse_loss = nn.MSELoss()(decoded, x)
        
        # Update metrics
        self.val_psnr(decoded, x)
        self.val_ssim(decoded, x)
        self.val_mae(decoded, x)
        self.val_lpips(decoded, x)

        # Log metrics
        self.log_dict({
            'val_loss': mse_loss,
            'val_psnr': self.val_psnr,
            'val_ssim': self.val_ssim,
            'val_mae': self.val_mae,
            'val_lpips': self.val_lpips
        }, on_epoch=True, prog_bar=True)

        # Log reconstructions only on the first batch
        if batch_idx == 0:
            # Check if WandB is disabled before running visualization code
            wandb_enabled = os.environ.get("WANDB_MODE", "").lower() != "disabled" and os.environ.get("WANDB_MODE", "").lower() != "dryrun"
            
            if wandb_enabled:
                self._log_reconstructions(x, decoded)
                
                # Store batch for further analysis
                self.last_val_batch = x
                self.last_val_decoded = decoded
                self.last_val_latent = z

        return mse_loss
        
    def on_validation_epoch_end(self):
        """Log additional metrics and visualizations at the end of validation epoch"""
        # Check if WandB is disabled
        wandb_enabled = os.environ.get("WANDB_MODE", "").lower() != "disabled" and os.environ.get("WANDB_MODE", "").lower() != "dryrun"
        
        if not wandb_enabled:
            return
        
        if not hasattr(self, 'last_val_batch'):
            return
            
        # Skip during sanity check
        if self.trainer.sanity_checking:
            return
            
        try:
            # Compute feature distribution in latent space
            latent_flat = self.last_val_latent.view(self.last_val_latent.size(0), -1)
            
            # Log latent space statistics
            self.log_dict({
                'latent/mean': latent_flat.mean().item(),
                'latent/std': latent_flat.std().item(),
                'latent/min': latent_flat.min().item(),
                'latent/max': latent_flat.max().item(),
                'latent/sparsity': (latent_flat.abs() < 0.01).float().mean().item()
            })
        except Exception as e:
            print(f"Error in on_validation_epoch_end: {e}")

    def _log_reconstructions(self, original, reconstructed):
        # Check if WandB logging is enabled
        wandb_enabled = os.environ.get("WANDB_MODE", "").lower() != "disabled" and os.environ.get("WANDB_MODE", "").lower() != "dryrun"
        
        # Skip WandB visualization if disabled
        if not wandb_enabled:
            return
            
        try:
            # Add masking visualization for first sample
            masked = self.apply_masking(original[:4])
            _, masked_rec, _ = self.model(masked)
            
            grid = torch.cat([
                original[:4].cpu(),
                masked.cpu(), 
                masked_rec.cpu(),
                reconstructed[:4].cpu()
            ], dim=0)
            
            grid = make_grid(grid, nrow=4, normalize=True)
            self.logger.experiment.log({
                "reconstructions/overview": wandb.Image(grid, caption="Top: Originals | Masked Inputs | Masked Reconstructions | Full Reconstructions"),
                "epoch": self.current_epoch
            })
            
            # Log individual samples with their reconstructions for detailed view
            for i in range(min(4, len(original))):
                sample_grid = torch.cat([
                    original[i].unsqueeze(0).cpu(),
                    reconstructed[i].unsqueeze(0).cpu()
                ], dim=0)
                sample_grid = make_grid(sample_grid, nrow=2, normalize=True)
                
                # Create temporary PSNR and SSIM metrics for per-sample calculation
                # Use the same device as the model
                sample_psnr_metric = PeakSignalNoiseRatio().to(self.device)
                sample_ssim_metric = StructuralSimilarityIndexMeasure().to(self.device)
                
                # Compute per-sample metrics
                sample_psnr = sample_psnr_metric(
                    reconstructed[i].unsqueeze(0), 
                    original[i].unsqueeze(0)
                ).item()
                
                sample_ssim = sample_ssim_metric(
                    reconstructed[i].unsqueeze(0), 
                    original[i].unsqueeze(0)
                ).item()
                
                self.logger.experiment.log({
                    f"reconstructions/sample_{i}": wandb.Image(
                        sample_grid, 
                        caption=f"Original vs Reconstructed (PSNR: {sample_psnr:.2f}, SSIM: {sample_ssim:.4f})"
                    )
                }, commit=False)
                
            # Log difference maps for error visualization
            diff_maps = torch.abs(original[:4] - reconstructed[:4])
            # Normalize difference maps for better visualization
            diff_maps = diff_maps / diff_maps.max() if diff_maps.max() > 0 else diff_maps
            
            diff_grid = make_grid(diff_maps.cpu(), nrow=4, normalize=True)
            self.logger.experiment.log({
                "reconstructions/error_maps": wandb.Image(diff_grid, caption="Reconstruction Error Maps")
            }, commit=False)
        except Exception as e:
            print(f"Error in _log_reconstructions: {e}")

    def configure_optimizers(self):
        # Get optimizer parameters from config if available
        lr = self.config.get('optimizer', {}).get('lr', 2e-4)
        weight_decay = self.config.get('optimizer', {}).get('weight_decay', 1e-4)
        betas = self.config.get('optimizer', {}).get('betas', (0.9, 0.95))
        
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            betas=betas
        )
        
        # Get scheduler parameters from config if available
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'onecycle')
        
        if scheduler_type == 'onecycle':
            max_lr = scheduler_config.get('max_lr', lr)
            pct_start = scheduler_config.get('pct_start', 0.3)
            
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=pct_start
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
            
        elif scheduler_type == 'cosine':
            T_max = scheduler_config.get('T_max', self.trainer.max_epochs)
            eta_min = scheduler_config.get('eta_min', 1e-6)
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=T_max,
                eta_min=eta_min
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
            
        elif scheduler_type == 'reduce_on_plateau':
            patience = scheduler_config.get('patience', 5)
            factor = scheduler_config.get('factor', 0.5)
            min_lr = scheduler_config.get('min_lr', 1e-6)
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=factor,
                patience=patience,
                min_lr=min_lr
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "monitor": "val_loss"}]
            
        else:
            # Default to OneCycleLR
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.3
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

# Load configuration from file
def load_config(config_path=None):
    """Load configuration from YAML file"""
    # Default config path if not specified
    if config_path is None and os.path.exists('histopathology/configs/wandb_config.yaml'):
        config_path = 'histopathology/configs/wandb_config.yaml'
        
    default_config = {
        'model': {
            'use_kan': True,
            'use_eca': True,
            'use_bam': True,
            'kan_options': {
                'kernel_size': [5, 5],
                'padding': [2, 2],
                'recon_kernel_size': [3, 3],
                'recon_padding': [1, 1]
            },
            'interpretability': {
                'enable_hooks': True,
                'enable_captum': True,
                'enable_gradcam': True,
                'store_activations': True
            }
        },
        'training': {
            'batch_size': 16,
            'max_epochs': 30,
            'precision': '16-mixed',
            'mask_ratio': 0.4,
            'noise_level': 0.3,
            'gradient_clip_val': 0.5,
            'accumulate_grad_batches': 2
        },
        'optimizer': {
            'lr': 2e-4,
            'weight_decay': 1e-4,
            'betas': [0.9, 0.95]
        },
        'scheduler': {
            'type': 'onecycle',
            'max_lr': 2e-4,
            'pct_start': 0.3
        },
        'wandb': {
            'project': 'histo-dae-robust',
            'entity': None,
            'name': f'dae-kan-robust-{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            'tags': ['autoencoder', 'kan', 'attention', 'histopathology'],
            'log_model': True,
            'log_artifacts': True,
            'log_code': True
        },
        'callbacks': {
            'early_stopping': {
                'monitor': 'val_psnr',
                'patience': 15,
                'mode': 'max'
            },
            'model_checkpoint': {
                'monitor': 'val_psnr',
                'save_top_k': 3,
                'mode': 'max'
            }
        },
        'advanced_logging': {
            'log_system_metrics': True,
            'log_system_freq': 100,
            'log_gradients': True,
            'log_gradients_freq': 100,
            'log_latent_freq': 500,
            'log_attention_freq': 1000,
            'log_gradcam_freq': 2000,
            'track_params': True,
            'track_cluster_evolution': True,
            'n_clusters': 5
        }
    }
    
    if config_path is not None and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            # Recursively update default config with loaded config
            def update_dict(d, u):
                for k, v in u.items():
                    if isinstance(v, dict) and k in d:
                        d[k] = update_dict(d[k], v)
                    else:
                        d[k] = v
                return d
            
            default_config = update_dict(default_config, config)
    
    return default_config

# Training configuration
def train_robust(config_path=None) -> None:
    """
    Train the robust denoising autoencoder model using curriculum learning.
    Sets up both training and validation datasets, configures training with
    mixed precision and gradient clipping.
    
    The function:
    - Loads configuration from file or uses defaults
    - Creates train and validation datasets
    - Initializes the RobustDAE model with masking and noise parameters
    - Configures Weights & Biases logging with advanced visualization
    - Sets up callbacks for model checkpointing, early stopping, and advanced logging
    - Runs training with PyTorch Lightning
    
    Args:
        config_path: Optional path to configuration file (YAML)
    """
    # Load configuration
    config = load_config(config_path)
    
    # Ensure specific settings are applied regardless of config file
    if 'wandb' in config and 'log_artifacts' in config['wandb']:
        config['wandb']['log_artifacts'] = False
    
    if 'callbacks' in config and 'model_checkpoint' in config['callbacks']:
        config['callbacks']['model_checkpoint']['save_top_k'] = 1
    
    # Extract training config
    training_config = config['training']
    wandb_config = config['wandb']
    advanced_logging_config = config['advanced_logging']
    
    # Initialize datasets
    X_train, y_train = create_dataset('train')
    X_val, y_val = create_dataset('test')  # Using test split for validation
    
    # Create datasets (pin_memory in DataLoader will handle GPU transfer)
    train_ds = ImageDataset(X_train, y_train)
    val_ds = ImageDataset(X_val, y_val)
    
    # Create dataloaders
    batch_size = training_config.get('batch_size', 16)
    num_workers = training_config.get('num_workers', 4)
    
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True  # Enable faster data transfer to GPU
    )
    
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True
    )
    
    # Initialize model
    model = RobustDAE(
        mask_ratio=training_config.get('mask_ratio', 0.4),
        noise_level=training_config.get('noise_level', 0.3),
        config=config,
        wandb_config=wandb_config
    )

    # Initialize WandB with enhanced config
    wandb_config['log_artifacts'] = False  # Always disable artifact logging
    run = wandb.init(
        project=wandb_config.get('project', 'histo-dae-robust'),
        entity=wandb_config.get('entity'),
        name=wandb_config.get('name', f'dae-kan-robust-{datetime.now().strftime("%Y%m%d-%H%M%S")}'),
        tags=wandb_config.get('tags', ['autoencoder', 'kan', 'attention', 'histopathology']),
        config=config  # Log full config for reproducibility
    )
    
    # Configure WandB logger
    wandb_logger = WandbLogger(
        experiment=run,
        log_model=wandb_config.get('log_model', True)
    )
    
    # Configure callbacks
    callbacks_config = config['callbacks']
    
    # Standard callbacks
    callbacks = [
        LearningRateMonitor(),
        ModelCheckpoint(
            monitor=callbacks_config.get('model_checkpoint', {}).get('monitor', 'val_psnr'),
            mode=callbacks_config.get('model_checkpoint', {}).get('mode', 'max'),
            save_top_k=1,  # Always save only the top 1 model
            filename='dae-kan-robust-{epoch:02d}-{val_psnr:.2f}'
        ),
        EarlyStopping(
            monitor=callbacks_config.get('early_stopping', {}).get('monitor', 'val_psnr'),
            patience=callbacks_config.get('early_stopping', {}).get('patience', 15),
            mode=callbacks_config.get('early_stopping', {}).get('mode', 'max')
        )
    ]
    
    # Add advanced WandB logging callback
    callbacks.append(
        AdvancedWandbCallback(**advanced_logging_config)
    )

    # Configure trainer with precision and gradient clipping
    trainer = pl.Trainer(
        max_epochs=training_config.get('max_epochs', 100),
        logger=wandb_logger,
        callbacks=callbacks,
        gradient_clip_val=training_config.get('gradient_clip_val', 0.5),
        accumulate_grad_batches=training_config.get('accumulate_grad_batches', 2),
        precision=training_config.get('precision', '16-mixed'),
        devices='auto',
        accelerator='auto'
    )

    # Log code for reproducibility if enabled
    if wandb_config.get('log_code', True):
        run.log_code(".")
        
    # Start training
    trainer.fit(model, train_dl, val_dl)
    
    # Close WandB run
    wandb.finish()


def create_wandb_config(path='histopathology/configs/wandb_config.yaml'):
    """Create a default WandB configuration file"""
    config = load_config()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Write config to file
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created default configuration at {path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DAE KAN Attention model')
    parser.add_argument('--config', type=str, default='histopathology/configs/wandb_config.yaml', 
                        help='Path to config file')
    parser.add_argument('--create-config', action='store_true', help='Create default config file')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_wandb_config()
    else:
        train_robust(args.config)
