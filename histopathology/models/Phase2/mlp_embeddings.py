import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import numpy as np
import pandas as pd
import wandb
import torchmetrics

# Define the PyTorch Lightning Module
class MLP(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        # Tambahkan Dropout 0.2
        # gunakan pretrained
        self.layer1 = nn.Sequential(nn.BatchNorm1d(input_size), nn.Linear(input_size, hidden_size))
        self.layer2 = nn.Sequential(nn.BatchNorm1d(hidden_size), nn.Linear(hidden_size, hidden_size // 2))
        self.layer3 = nn.Sequential(nn.BatchNorm1d(hidden_size // 2), nn.Linear(hidden_size // 2, hidden_size // 2))
        self.layer4 = nn.Linear(hidden_size // 2, num_classes)
        self.relu = nn.ELU()
        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()

         # Metrics
        self.train_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.train_f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='macro')
        self.val_f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='macro')


    def forward(self, x):
        x = self.relu(self.layer1(x))
        embedding1 = self.relu(self.layer2(x))
        embedding2 = self.relu(self.layer3(embedding1))
        out = self.softmax(self.relu(self.layer4(embedding2)))
        concatenated_embeddings = torch.cat((x, embedding1, embedding2), dim=1)
        return out, concatenated_embeddings

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        logits, _ = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, y)
        self.train_f1(preds, y)

        self.log('train_loss', loss)
        self.log('train_acc', self.train_accuracy, on_step=False, on_epoch=True)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        logits, _ = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.val_accuracy(preds, y)
        self.val_f1(preds, y)

        self.log('val_loss', loss)
        self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        return optimizer


def extract_embeddings(model, dataloader, device='cuda'):
    model.to(device)
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            data, label = batch
            data, label = data.to(device), label.to(device)
            _, embedding = model(data)
            embeddings.append(embedding.cpu().numpy())
            labels.append(label.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

    return embeddings, labels

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data, labels, device='cpu'):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].to(self.device), self.labels[idx].to(self.device)


print('Create Dataset')

# Load and preprocess the data
file_path = '../../data/processed/DAE_Embeddings.csv'
data = pd.read_csv(file_path)

# Assuming the last column is 'class'
X = data.drop(columns=['label']).values
y = data['label'].values - 1

dataset = CustomDataset(X, y, 'cuda')

train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=0.1, random_state=999, shuffle=True, stratify=y)

train_set = Subset(dataset, train_idx)
valid_set = Subset(dataset, val_idx)

print('Loading Dataset')
# Split the dataset into training and validation sets

# TO-DO: Fix Data Splitting
train_dataset = Subset(dataset, train_idx)
valid_dataset = Subset(dataset, val_idx)

print('Loading Dataset')
train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
val_loader = DataLoader(valid_dataset, batch_size=4096)

# Instantiate the model
input_size = 4096
hidden_size = 2048
num_classes = 14

print("Training Model")
model = MLP(input_size, hidden_size, num_classes).to('cuda')
wandb_logger = WandbLogger(project='mlp-histopathology')

wandb_logger.watch(model, log='all', log_freq=10)

lr_monitor = LearningRateMonitor(logging_interval='epoch')
checkpoint_callback = ModelCheckpoint(monitor='val_loss')

# Initialize the trainer
trainer = pl.Trainer(
    max_epochs=500,
    logger=wandb_logger,
    callbacks=[lr_monitor, checkpoint_callback]
)

trainer.fit(model, train_loader, val_loader)


print('Saving Model')
# Save the trained model
model_path = 'trained_model/MLP/mlp_model.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")


print("Extracting embeddings")
# Extract embeddings for the entire dataset
embeddings, labels = extract_embeddings(model, DataLoader(dataset, batch_size=8192, shuffle=True))

# Save embeddings to CSV
embeddings_df = pd.DataFrame(embeddings)
embeddings_df['label'] = labels
embeddings_df.to_csv('../../data/processed/mlp_embeddings.csv', index=False)

print("Embeddings saved to processed")
