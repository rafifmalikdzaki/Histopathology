from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
import torchmetrics
from model import DAE_KAN_Attention
from histopathology_dataset import *
from torch.utils.data import Dataset, DataLoader
import wandb

class MyModel(pl.LightningModule):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = DAE_KAN_Attention()
        
        # Initialize metrics
        self.train_psnr = torchmetrics.PSNR()
        self.train_ssim = torchmetrics.StructuralSimilarityIndexMeasure()
        self.val_psnr = torchmetrics.PSNR()
        self.val_ssim = torchmetrics.StructuralSimilarityIndexMeasure()
        self.val_mae = torchmetrics.MeanAbsoluteError()

    def forward(self, x):
        encoded, decoded, z = self.model(x)
        return encoded, decoded, z

    def training_step(self, batch, batch_idx):
        x, _ = batch
        encoded, decoded, z = self.forward(x)

        # MSE loss between input and reconstruction
        mse_loss = nn.MSELoss()(x, decoded)

        # KAN regularization loss
        reg_loss = self.model.regularization_loss()
        
        # Total loss with regularization
        loss = mse_loss + reg_loss * 0.5  # Scale regularization

        # Update metrics
        self.train_psnr(decoded, x)
        self.train_ssim(decoded, x)
        
        # Logging losses and metrics
        self.log_dict({
            'train_loss': loss,
            'train_mse_loss': mse_loss,
            'train_reg_loss': reg_loss,
            'train_psnr': self.train_psnr,
            'train_ssim': self.train_ssim,
            'learning_rate': self.trainer.optimizers[0].param_groups[0]['lr']
        }, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        encoded, decoded, z = self.forward(x)

        # MSE loss between input and reconstruction
        mse_loss = nn.MSELoss()(x, decoded)

        # Total loss
        loss = mse_loss

        # Update metrics
        self.val_psnr(decoded, x)
        self.val_ssim(decoded, x)
        self.val_mae(decoded, x)
        
        # Log metrics
        self.log_dict({
            'val_loss': loss,
            'val_mse_loss': mse_loss,
            'val_psnr': self.val_psnr,
            'val_ssim': self.val_ssim,
            'val_mae': self.val_mae
        }, on_epoch=True, prog_bar=True)
        
        # Log sample reconstructions
        if batch_idx == 0:
            self._log_reconstructions(x, decoded)

        return loss

    def _log_reconstructions(self, original, reconstructed):
        # Log first 4 samples from batch
        images = []
        for i in range(min(4, original.shape[0])):
            img_pair = torch.cat([
                original[i].cpu(), 
                reconstructed[i].cpu()
            ], dim=-1)
            images.append(wandb.Image(img_pair, caption=f"Sample {i}"))
        
        self.logger.experiment.log({
            "Reconstructions": images,
            "epoch": self.current_epoch
        })

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=30, eta_min=1e-5
        )
        return [optimizer], [scheduler]

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create datasets and dataloaders
    train_ds = ImageDataset(*create_dataset('train'), device)
    test_ds = ImageDataset(*create_dataset('test'), device)

    train_dl = DataLoader(train_ds, batch_size=12, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=12, shuffle=True)
    x, y = next(iter(train_dl))
    # print(x.shape, y.shape)
    model = MyModel()

    # Set up WandbLogger
    wandb_logger = WandbLogger(project='histo-dae')

    wandb_logger.watch(model, log='all', log_freq=10)

    # Set up callbacks
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss')

    # Initialize the trainer
    trainer = pl.Trainer(
        max_epochs=100,
        logger=wandb_logger,
        callbacks=[lr_monitor, checkpoint_callback],
        gradient_clip_val=1.0,
        accumulate_grad_batches=2
    )

    # # Train the model
    trainer.fit(model, train_dl, test_dl)

    # Finish the wandb run
    wandb.finish()
