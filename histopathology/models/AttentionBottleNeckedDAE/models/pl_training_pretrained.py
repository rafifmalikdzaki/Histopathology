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
        x, _ = batch
        encoded, decoded, z = self.forward(x)

        # MSE loss between input and reconstruction
        mse_loss = nn.MSELoss()(x, decoded)
        # KL divergence loss
        kl_div = -0.5 * torch.sum(1 + encoded - encoded.pow(2) - torch.exp(encoded))

        # Total loss
        loss = mse_loss + kl_div

        # Logging losses
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_mse_loss', mse_loss, on_epoch=True)
        self.log('train_kl_div', kl_div, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        encoded, decoded, z = self.forward(x)

        # MSE loss between input and reconstruction
        mse_loss = nn.MSELoss()(x, decoded)
        # KL divergence loss
        kl_div = -0.5 * torch.sum(1 + encoded - encoded.pow(2) - torch.exp(encoded))

        # Total loss
        loss = mse_loss + kl_div

        # Logging losses
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_mse_loss', mse_loss, on_epoch=True)
        self.log('val_kl_div', kl_div, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5,
                                                               min_lr=1e-6)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }


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
image_dir = './data/processed/PANnuke/'

# Create the dataset
dataset = ImageDataset(image_dir=image_dir)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model = MyModel()

# Set up WandbLogger
wandb_logger = WandbLogger(project='histo-dae')

wandb_logger.watch(model, log='all', log_freq=10)

# Set up callbacks
early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=True, mode='min')
lr_monitor = LearningRateMonitor(logging_interval='epoch')
checkpoint_callback = ModelCheckpoint(monitor='val_loss')

# Initialize the trainer
trainer = pl.Trainer(
    max_epochs=10,
    logger=wandb_logger,
    callbacks=[early_stop_callback, lr_monitor, checkpoint_callback]
)

# # Train the model
trainer.fit(model, train_dataloaders=dataloader)

# Finish the wandb run
wandb.finish()
