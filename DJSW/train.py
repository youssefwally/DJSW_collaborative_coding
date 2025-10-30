import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.wmlp import WMLP
from utils.wdataloader import USPS06Dataset
from models.smlp import SMLP
from utils.mnist_dataset import MnistH5Dataset
from models.dmlp import DMLP
from utils.dataset_mnist03_h5 import Mnist03Dataset

from torch.utils.data import random_split
from pathlib import Path

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        # If inputs have spatial dimensions (e.g. 1x28x28), flatten to (B, C*H*W)
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), -1)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            if inputs.dim() > 2:
                inputs = inputs.view(inputs.size(0), -1)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train_pipeline(img_dim, train_loader, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    if args.username == "waly":
        model = WMLP(img_dim, 7).to(device)
    elif args.username == "sigurd":
        # Reduced MNIST digits 4..9 -> remap to 0..5 in the dataset creation step
        # img_dim is passed in (should be 28*28)
        model = SMLP(input_size=img_dim, hidden_size=77, output_size=6).to(device)
    elif args.username == "dennis":
        model = DMLP(input_dim=img_dim, output_dim=4, hidden_dim=300, negative_slope=0.01).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    for epoch in range(args.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{args.output_dir}/{args.exp_name}_checkpoint_epoch_{epoch+1}.pt")


def train_model(args):
    # Create datasets
    if args.username == "waly":
        train_dataset = USPS06Dataset(set_type="train")
        val_dataset = USPS06Dataset(set_type="val")
        img_dim = train_dataset.get_input_dim()
    elif args.username == "sigurd":
        # Use reduced MNIST HDF5 (digits 4..9). We remap targets to 0-5
        h5_path = "data/processed/mnist_4_9.h5"
        full_ds = MnistH5Dataset(h5_path, target_transform=lambda t: int(t) - 4)
        # split into train/val (90/10)
        val_size = int(len(full_ds) * 0.1)
        train_size = len(full_ds) - val_size
        train_dataset, val_dataset = random_split(full_ds, [train_size, val_size])
        img_dim = 28 * 28
    elif args.username == "dennis":
        ROOT = Path(__file__).resolve().parents[1]
        H5_DIR = ROOT / "data" / "processed"
        H5_DIR.mkdir(parents=True, exist_ok=True)
        train_dataset = Mnist03Dataset(h5_path=H5_DIR / "mnist03.h5", split="train")
        val_dataset = Mnist03Dataset(h5_path=H5_DIR / "mnist03.h5", split="val")
        img_dim = 784
    
    

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Image dimensions: {img_dim}")

    train_pipeline(img_dim, train_loader, val_loader, args)