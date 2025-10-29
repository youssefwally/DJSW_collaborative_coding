import os
import h5py
import numpy as np
import torch
from pathlib import Path

class Mnist03Dataset:
    """
    MNIST03 dataset stored in an HDF5 file:
      /<split>/images : uint8 [N, 28, 28]
      /<split>/labels : int64 [N]
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
        """Open the HDF5 file in this process if needed (one handle per process)."""
        pid = os.getpid()
        if (self._file is None) or (self._pid != pid):
            # Close old handle if coming from a different process
            if self._file is not None:
                try: self._file.close()
                except Exception: pass
            self._file = h5py.File(self.h5_path, "r")  # read-only
            self._pid = pid

        return self._file

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
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
        try:
            if self._file is not None:
                self._file.close()
        except Exception:
            pass