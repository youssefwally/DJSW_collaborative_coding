"""HDF5-backed MNIST Dataset for PyTorch.

This implements a lightweight dataset that lazily opens an HDF5 file and
returns (image_tensor, label) pairs. Images are returned as uint8 tensors
scaled to [0,1] if a transform isn't provided.
"""
from pathlib import Path
from typing import Optional, Callable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class MnistH5Dataset(Dataset):
    def __init__(self, h5_path: str | Path, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")
        self.transform = transform
        self.target_transform = target_transform
        self._h5 = None

    def _ensure_open(self):
        # Open file lazily to be compatible with multi-worker DataLoader
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    def __len__(self):
        self._ensure_open()
        return int(self._h5["labels"].shape[0])

    def __getitem__(self, idx):
        self._ensure_open()
        img = self._h5["images"][idx]  # uint8 H x W
        label = int(self._h5["labels"][idx])

        # Convert to tensor
        img = torch.from_numpy(np.asarray(img))
        # Ensure channel dimension (C, H, W)
        if img.ndim == 2:
            img = img.unsqueeze(0)
        # Convert dtype to float and scale to [0,1]
        img = img.float().div(255.0)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def close(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass
            self._h5 = None

    def __del__(self):
        self.close()
