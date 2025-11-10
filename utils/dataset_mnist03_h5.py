import os
import h5py
import numpy as np
import torch
from pathlib import Path

class Mnist03Dataset:
    """
    Dataset class for MNIST03 (MNIST numbers 0-3) stored in HDF5 format.
    Split in train, val and test set, image data and corresponding labels for each split are accessible through
      /<split>/images : uint8 [N, 28, 28]
      /<split>/labels : int64 [N]

      Args:
            h5_path: Path to HDF5 file
            split: Dataset split ("train", "val" or "test"). Default is "train".
            normalize: bool
                If True, normalize the images to [0, 1]. Default is True.
            flatten: bool
                If True, flatten the images from (1, 28, 28) to (784,). Default is True.
    """
    def __init__(self, h5_path, split="train", normalize=True, flatten=True):
        self.h5_path = str(Path(h5_path))
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train' or 'val' or 'test'")
        self.split = split
        self.normalize = normalize
        self.flatten = flatten

        self._file = None      # h5py.File handle (lazy-open)
        self._pid = None       # track process id to re-open after fork
        self._length = None    # cached length

        # Peek length without keeping file open
        with h5py.File(self.h5_path, "r") as f:
            self._length = f[f"{self.split}/labels"].shape[0]

    def _ensure_open(self):
        """
        Open the HDF5 file

        Returns: h5py.File
            A handle to the open HDF5 file (read-only)
        """
        pid = os.getpid()
        if (self._file is None) or (self._pid != pid):
            # Close old handle if coming from a different process
            if self._file is not None:
                try: self._file.close()
                except Exception: pass
            self._file = h5py.File(self.h5_path, "r")
            self._pid = pid

        return self._file

    def __len__(self):
        """
        Number of samples in the current split.
        """
        return self._length

    def __getitem__(self, idx):
        """
        Get a sample from the data set
        Args:
            idx: index of sample

        Returns:
            X_t: torch.Tensor
                sample image tensor
            y_t: torch.Tensor
                sample label tensor
        """
        f = self._ensure_open()
        X = f[f"{self.split}/images"][idx]   # shape (28, 28), uint8 from disk
        y = int(f[f"{self.split}/labels"][idx])

        # to float32
        X = X.astype("float32")
        if self.normalize:
            X /= 255.0

        # (1, 28, 28)
        X = np.expand_dims(X, axis=0)

        if self.flatten:
            X = X.reshape(-1)  # (784,)

        X_t = torch.from_numpy(X)                 # float32
        y_t = torch.tensor(y, dtype=torch.long)   # int64
        return X_t, y_t

    def __del__(self):
        """
        Close the HDF5 file
        """
        try:
            if self._file is not None:
                self._file.close()
        except Exception:
            pass