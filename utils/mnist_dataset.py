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
        """Initialize the HDF5-backed MNIST dataset.

        Parameters
        ----------
        h5_path:
            Path to an HDF5 file containing two datasets: ``images`` and ``labels``.
            The constructor will raise FileNotFoundError if the file does not exist.
        transform:
            Optional image transform (callable) applied to the image tensor after
            it has been converted to float and scaled to [0, 1]. Typical use is a
            torchvision transform or a custom preprocessing function.
        target_transform:
            Optional callable applied to the label (useful to remap labels on-the-fly,
            e.g. ``lambda t: int(t)-4`` to map 4..9 -> 0..5).

        Notes
        -----
        The dataset lazily opens the HDF5 file when data is first accessed. This
        is important when using ``torch.utils.data.DataLoader`` with multiple
        workers: each worker will get its own file handle via the lazy opener.
        """
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")
        self.transform = transform
        self.target_transform = target_transform
        self._h5 = None

    def _ensure_open(self):
        # Open file to be compatible with multi-worker DataLoader
        if self._h5 is None:
            # Keep the file open for the lifetime of the Dataset instance or
            # the worker process. Closing is performed in ``close()`` / ``__del__``.
            self._h5 = h5py.File(self.h5_path, "r")

    def __len__(self):
        """Return the number of samples in the dataset.

        This reads the ``labels`` dataset's first dimension to determine
        the length and therefore triggers lazy opening of the HDF5 file.
        """
        self._ensure_open()
        return int(self._h5["labels"].shape[0])

    def __getitem__(self, idx):
        """Return the (image, label) pair for the given index.

        Parameters
        ----------
        idx : int
            Index of the requested sample.

        Returns
        -------
        (torch.Tensor, int)
            A tuple of (image_tensor, label). The image tensor is a float
            tensor scaled to [0,1] with shape (C, H, W). Label is an int and
            may have been transformed by ``target_transform`` if provided.

        Raises
        ------
        IndexError
            If the index is out of bounds for the dataset.
        """
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
        """Close the underlying HDF5 file handle if open.

        This is safe to call multiple times. It is also invoked from
        ``__del__`` to ensure resources are released when the dataset is
        garbage-collected.
        """
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                # Best-effort close; do not raise during object destruction.
                pass
            self._h5 = None

    def __del__(self):
        # Ensure the file is closed when the object is destroyed.
        try:
            self.close()
        except Exception:
            # Avoid raising exceptions during interpreter shutdown.
            pass
