
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import nibabel as nib  # Library to load .nii files
import sys
import os
# Add the parent directory of the current script to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.Unet_PyTorch import UNet3D

class MRIDataset(Dataset):
    def __init__(self, lr_dir, hr_dir):
        # Get all .nii files in the low-res and high-res directories
        self.lr_files = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir) if f.endswith('.nii')])
        self.hr_files = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir) if f.endswith('.nii')])

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        # Load low-resolution and high-resolution images from .nii files
        lr_img = nib.load(self.lr_files[idx]).get_fdata()
        hr_img = nib.load(self.hr_files[idx]).get_fdata()

        # Convert NIfTI images to PyTorch tensors
        lr_img = torch.from_numpy(lr_img).float().unsqueeze(0)  # Add channel dimension
        hr_img = torch.from_numpy(hr_img).float().unsqueeze(0)  # Add channel dimension

        return lr_img, hr_img

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for lr, hr in tqdm(dataloader):
        lr, hr = lr.to(device), hr.to(device)

        optimizer.zero_grad()

        outputs = model(lr)
        loss = criterion(outputs, hr)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * lr.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for lr, hr in dataloader:
            lr, hr = lr.to(device), hr.to(device)

            outputs = model(lr)
            loss = criterion(outputs, hr)

            running_loss += loss.item() * lr.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def train(model, train_loader, val_loader, num_epochs, criterion, optimizer, device):
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Save model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Saved model with validation loss: {val_loss:.4f}')

def main():
    # Paths to the low-res and high-res data
    lr_dir = 'data/train/low_res'
    hr_dir = 'data/train/high_res'

    # Dataset and DataLoader
    train_dataset = MRIDataset(lr_dir, hr_dir)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

    # Use 10% of the training data for validation
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    # Model, criterion, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3D().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    num_epochs = 100
    train(model, train_loader, val_loader, num_epochs, criterion, optimizer, device)

if __name__ == '__main__':
    main()