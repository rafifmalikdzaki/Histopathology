import torch
import torch.utils.data
import pandas as pd
import numpy as np
from model import DAE_KAN_Attention
from histopathology_dataset import create_dataset, ImageDataset
from tqdm import tqdm

def get_dae_embedding(model, checkpoint, dataloader, device='cuda'):
    # Load checkpoint and update state dict
    new_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        new_key = key.replace("model.", "")
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    model.eval()

    embeddings = []
    labels = []

    with torch.no_grad():
        for data, label in tqdm(dataloader):
            data = data.to(device)
            _, _, z = model(data)
            z = z.view(z.size(0), -1)
            embeddings.append(z.cpu().numpy())
            labels.append(label.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

    return embeddings, labels

# Path to the checkpoint
checkpoint_path = "./histo-dae/Finetuned/checkpoints/epoch=29-step=24415.ckpt"
checkpoint = torch.load(checkpoint_path)

# Create dataset and dataloader
data = create_dataset('OHEheparfix')
train_ds = ImageDataset(*data, 'cuda')
dataloader = torch.utils.data.DataLoader(train_ds, batch_size=192)

# Instantiate the original model
original_model = DAE_KAN_Attention().to('cuda')

# Get the embeddings and labels
embeddings, labels = get_dae_embedding(original_model, checkpoint, dataloader, device='cuda')

# Save to CSV
image_path = f"./data/processed/DAE_Embeddings.csv"
df = pd.DataFrame(embeddings)
df['label'] = labels
df.to_csv(image_path, index=False)

print(f"Embeddings shape: {embeddings.shape}, Labels shape: {labels.shape}")
print("Embeddings saved to embeddings.csv")
