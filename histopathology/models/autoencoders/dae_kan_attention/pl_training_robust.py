from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
import torchmetrics
from model import DAE_KAN_Attention
from histopathology_dataset import *
import wandb
import torchvision.transforms as T
from torchvision.utils import make_grid

class RobustDAE(pl.LightningModule):
    def __init__(self, mask_ratio=0.3, noise_level=0.2):
        super().__init__()
        self.model = DAE_KAN_Attention()
        self.mask_ratio = mask_ratio
        self.noise_level = noise_level
        
        # Initialize metrics
        self.train_psnr = torchmetrics.PSNR()
        self.train_ssim = torchmetrics.StructuralSimilarityIndexMeasure()
        self.val_psnr = torchmetrics.PSNR()
        self.val_ssim = torchmetrics.StructuralSimilarityIndexMeasure()
        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.val_lpips = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarness(net_type='vgg')

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

    def training_step(self, batch, batch_idx):
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
        
        # Logging
        self.log_dict({
            'train_loss': loss,
            'train_mse_loss': mse_loss,
            'train_reg_loss': reg_loss,
            'train_psnr': self.train_psnr,
            'train_ssim': self.train_ssim,
            'curriculum_weight': curriculum_weight,
            'learning_rate': self.trainer.optimizers[0].param_groups[0]['lr']
        }, on_epoch=True, prog_bar=True)

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

        # Log reconstructions
        if batch_idx == 0:
            self._log_reconstructions(x, decoded)

        return mse_loss

    def _log_reconstructions(self, original, reconstructed):
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
            "Reconstructions": wandb.Image(grid, caption="Top: Originals | Masked Inputs | Masked Reconstructions | Full Reconstructions"),
            "epoch": self.current_epoch
        })

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=2e-4, 
            weight_decay=1e-4,
            betas=(0.9, 0.95)
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=2e-4,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.3
        )
        return [optimizer], [scheduler]

# Training configuration
def train_robust():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize dataset and model
    train_ds = ImageDataset(*create_dataset('train'), device)
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, persistent_workers=True)
    
    model = RobustDAE(mask_ratio=0.4, noise_level=0.3)

    # Configure logger and callbacks
    wandb_logger = WandbLogger(project='histo-dae-robust', log_model=True)
    callbacks = [
        LearningRateMonitor(),
        ModelCheckpoint(monitor='val_psnr', mode='max', save_top_k=3),
        EarlyStopping(monitor='val_psnr', patience=15, mode='max')
    ]

    # Configure trainer with precision and gradient clipping
    trainer = pl.Trainer(
        max_epochs=100,
        logger=wandb_logger,
        callbacks=callbacks,
        gradient_clip_val=0.5,
        accumulate_grad_batches=2,
        precision='16-mixed',
        devices='auto',
        accelerator='auto'
    )

    trainer.fit(model, train_dl)

if __name__ == '__main__':
    train_robust()
