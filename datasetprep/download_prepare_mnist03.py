import numpy as np
from pathlib import Path
import urllib.request
import gzip
import struct
from sklearn.model_selection import train_test_split
import h5py

ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = ROOT / "data" / "download"
NPZ_DIR = ROOT / "data" / "npz"

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
NPZ_DIR.mkdir(parents=True, exist_ok=True)

orig = {
    "train_imgs": "train-images-idx3-ubyte.gz",
    "train_anns": "train-labels-idx1-ubyte.gz",
    "test_imgs": "t10k-images-idx3-ubyte.gz",
    "test_anns": "t10k-labels-idx1-ubyte.gz"
}

def load_idx_images(path: Path):
    """Load MNIST image file (idx3 format) from .gz"""
    with gzip.open(path, "rb") as f:
        _, n_images, n_rows, n_cols = struct.unpack(">IIII", f.read(16))
        # > means big-endian, I = 4-byte unsigned int
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n_images, n_rows, n_cols)


def load_idx_labels(path: Path):
    """Load MNIST label file (idx1 format) from .gz"""
    with gzip.open(path, "rb") as f:
        f.read(8)
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data

def save_to_npz(npz_path, data):
    np.savez_compressed(
        npz_path,
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_val=["X_val"],
        y_val=["y_val"],
        X_test=["X_test"],
        y_test=["y_test"],
    )
    print(f"mnist03 data saved under {npz_path}.")

def save_to_h5(h5_path, data):
    h5_path = Path(h5_path)
    # Choose chunk size roughly like your batch size
    chunk = (32, 28, 28)  # (N, H, W)

    with h5py.File(h5_path, "w") as f:
        for split in ("train", "val", "test"):
            X = data[f"X_{split}"]          # shape (N, 28, 28)
            y = data[f"y_{split}"].astype(np.int64)

            # Store images
            f.create_dataset(
                f"{split}/images",
                data=X,
                dtype="uint8",
                chunks=chunk,
                compression="gzip",
                shuffle=True,
            )
            # Store labels
            f.create_dataset(
                f"{split}/labels",
                data=y,
                dtype="int64",
                chunks=(min(len(y), 8192),),   # 1D labels; chunk as you like
                compression="gzip",
                shuffle=True,
            )
    print(f"mnist03 data saved under {h5_path}.")

def main(npz=False, h5=True):
    for _, value in orig.items():
        urllib.request.urlretrieve("https://storage.googleapis.com/cvdf-datasets/mnist/" + value, DOWNLOAD_DIR / value)
    print("MNIST original data download completed.")

    train_images = load_idx_images(DOWNLOAD_DIR / orig["train_imgs"])
    train_anns = load_idx_labels(DOWNLOAD_DIR / orig["train_anns"])
    test_images = load_idx_images(DOWNLOAD_DIR / orig["test_imgs"])
    test_anns = load_idx_labels(DOWNLOAD_DIR / orig["test_anns"])

    train_images_filtered = train_images[(train_anns==0) | (train_anns==1) | (train_anns==2) | (train_anns==3)]
    train_anns_filtered = train_anns[(train_anns==0) | (train_anns==1) | (train_anns==2) | (train_anns==3)]
    X_test = test_images[(test_anns==0) | (test_anns==1) | (test_anns==2) | (test_anns==3)]
    y_test = test_anns[(test_anns==0) | (test_anns==1) | (test_anns==2) | (test_anns==3)]
    X_train, X_val, y_train, y_val = train_test_split(train_images_filtered, train_anns_filtered, test_size=X_test.shape[0], random_state=42)

    data = dict(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

    if npz:
        save_to_npz(NPZ_DIR / "mnist03.npz",data)
    if h5:
        save_to_h5(NPZ_DIR / "mnist03.h5",data)

if __name__ == "__main__":
    main(npz=True, h5=True)