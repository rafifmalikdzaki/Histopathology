from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from model import DAE_KAN_Attention
from histopathology_dataset import *
from torch.utils.data import Dataset, DataLoader
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

        # Total loss
        loss = mse_loss

        # Logging losses
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_mse_loss', mse_loss, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        encoded, decoded, z = self.forward(x)

        # MSE loss between input and reconstruction
        mse_loss = nn.MSELoss()(x, decoded)

        # Total loss
        loss = mse_loss

        # Logging losses
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_mse_loss', mse_loss, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-6)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

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
        max_epochs=30,
        logger=wandb_logger,
        callbacks=[lr_monitor, checkpoint_callback]
    )

    # # Train the model
    trainer.fit(model, train_dl, test_dl)

    # Finish the wandb run
    wandb.finish()