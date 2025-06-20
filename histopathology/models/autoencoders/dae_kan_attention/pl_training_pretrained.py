from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from model import DAE_KAN_Attention
from histopathology_dataset import *
import wandb


class MyModel(pl.LightningModule):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = DAE_KAN_Attention()

    def forward(self, x):
        encoded, decoded, z = self.model(x)
        return encoded, decoded, z

    def training_step(self, batch, batch_idx):
        x = batch
        encoded, decoded, z = self.forward(x)

        # Calculate losses
        mse_loss = nn.MSELoss()(x, decoded)
        reg_loss = self.model.regularization_loss()
        loss = mse_loss + reg_loss * 0.5

        # Log metrics
        self.log_dict({
            'train_loss': loss,
            'train_mse_loss': mse_loss,
            'train_reg_loss': reg_loss,
            'learning_rate': self.trainer.optimizers[0].param_groups[0]['lr']
        }, on_epoch=True, prog_bar=True)

        # Log gradients
        if batch_idx % 100 == 0:
            for name, param in self.named_parameters():
                if param.grad is not None:
                    self.logger.experiment.log({
                        f"gradients/{name}": wandb.Histogram(param.grad.cpu().numpy())
                    })

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        encoded, decoded, z = self.forward(x)

        # Calculate losses
        mse_loss = nn.MSELoss()(x, decoded)
        loss = mse_loss

        # Calculate image metrics
        psnr = 10 * torch.log10(1 / mse_loss)
        ssim = torchmetrics.functional.structural_similarity_index_measure(decoded, x)

        # Log metrics
        self.log_dict({
            'val_loss': loss,
            'val_mse_loss': mse_loss,
            'val_psnr': psnr,
            'val_ssim': ssim
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-6
        )
        return [optimizer], [scheduler]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Create datasets and dataloaders
# train_ds = ImageDataset(*create_dataset('train'), device)
# test_ds = ImageDataset(*create_dataset('test'), device)
#
# train_dl = torch.utils.data.DataLoader(train_ds, batch_size=8)
# test_dl = torch.utils.data.DataLoader(test_ds, batch_size=8)
# x, y = next(iter(train_dl))
# print(x.shape, y.shape)

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, image_dir):
        """
        Args:
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = torchvision.io.read_image(img_name)
        image = image.type(torch.float32) / 255

        return image.to('cuda')


# Define the image directory
image_dir = './data/processed/HeparUnifiedPNG/'

# Create the dataset
dataset = ImageDataset(image_dir=image_dir)
dataloader = DataLoader(dataset, batch_size=12, shuffle=True)

model = MyModel()

# Set up WandbLogger with config
wandb_logger = WandbLogger(
    project='histo-dae',
    log_model='all',
    save_dir='./wandb',
    config={
        "batch_size": 12,
        "learning_rate": 2e-4,
        "weight_decay": 1e-4,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR"
    }
)

# Log model architecture and gradients
wandb_logger.watch(
    model, 
    log='all',
    log_freq=100,
    log_graph=True
)

# Set up callbacks
lr_monitor = LearningRateMonitor(logging_interval='epoch')
checkpoint_callback = ModelCheckpoint(monitor='train_loss')

# Initialize trainer with more callbacks
trainer = pl.Trainer(
    max_epochs=5,
    logger=wandb_logger,
    callbacks=[
        lr_monitor,
        checkpoint_callback,
        pl.callbacks.ProgressBar(refresh_rate=50)
    ],
    log_every_n_steps=50,
    accelerator='auto',
    devices='auto'
)

# # Train the model
trainer.fit(model, train_dataloaders=dataloader)

# Finish the wandb run
wandb.finish()
