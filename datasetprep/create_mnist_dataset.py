"""Download MNIST and store as HDF5.

Creates a dataset file at data/processed/mnist.h5 with two datasets:
 - images: uint8, shape (N, 28, 28)
 - labels: uint8, shape (N,)

Provides function create_mnist_h5(output_path, train=True, download=True)
and a simple CLI.
"""
from pathlib import Path
import h5py
import numpy as np

try:
    from torchvision.datasets import MNIST
    from torchvision import transforms
except Exception:
    raise RuntimeError("torchvision is required to download MNIST. Install torchvision.")


def create_mnist_h5(output_path: str | Path = "data/processed/mnist.h5", train: bool = True, download: bool = True, labels_to_keep: list[int] | None = None):
    """Create an HDF5 file containing MNIST images and labels.

    Parameters
    ----------
    output_path:
        Path to the output HDF5 file. Parent directories will be created if
        they do not exist.
    train:
        If True, use the MNIST training split; if False, use the test split.
    download:
        If True allow torchvision to download MNIST if the raw files are absent.
    labels_to_keep:
        Optional list of integer labels to include in the output. If provided,
        only samples whose labels are in this list are written into the HDF5
        file. Note that stored labels remain the original MNIST labels (e.g.
        4..9); any remapping for training should be applied at load time via
        ``target_transform``.

    Returns
    -------
    Path
        The path to the created HDF5 file.

    Raises
    ------
    RuntimeError
        If torchvision is not available (checked at module import time).

    Notes
    -----
    The function writes two datasets to the HDF5 file:
    - ``images``: uint8 array with shape (N, 28, 28)
    - ``labels``: uint8 array with shape (N,)

    The datasets use gzip compression to reduce disk usage.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Download using torchvision into a temporary folder under data/raw
    raw_dir = Path("data/raw/mnist")
    raw_dir.mkdir(parents=True, exist_ok=True)

    ds = MNIST(root=str(raw_dir), train=train, download=download, transform=None)

    # ds.data is a torch tensor (N, 28, 28), ds.targets is (N,)
    try:
        imgs = ds.data.numpy()
        labels = ds.targets.numpy()
    except Exception:
        # Older torchvision sometimes stores as PIL images; handle that
        imgs = np.stack([np.array(img) for img in ds.data], axis=0)
        labels = np.array(ds.targets)

    # Ensure dtype uint8 and shape (N,28,28)
    imgs = imgs.astype(np.uint8)
    labels = labels.astype(np.uint8)

    # Filter labels if requested
    if labels_to_keep is not None:
        labels_to_keep = list(labels_to_keep)
        mask = np.isin(labels, labels_to_keep)
        imgs = imgs[mask]
        labels = labels[mask]

    with h5py.File(output_path, "w") as f:
        f.create_dataset("images", data=imgs, compression="gzip")
        f.create_dataset("labels", data=labels, compression="gzip")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create MNIST HDF5 file")
    parser.add_argument("--output", "-o", default="data/processed/mnist.h5", help="Output HDF5 path")
    parser.add_argument("--train", action="store_true", help="Download train split (default)")
    parser.add_argument("--test", dest="train", action="store_false", help="Download test split")
    parser.add_argument("--keep", "--labels-to-keep", dest="labels_to_keep", default=None,
                        help="Comma-separated list of integer labels to keep (e.g. 4,5,6,7,8,9)."
                        )
    parser.set_defaults(train=True)
    args = parser.parse_args()
    # Parse comma-separated labels list into a Python list of ints if provided
    if args.labels_to_keep is None:
        labels_to_keep = None
    else:
        labels_to_keep = [int(x) for x in args.labels_to_keep.split(",") if x.strip() != ""]

    p = create_mnist_h5(args.output, train=args.train, labels_to_keep=labels_to_keep)
    print(f"Created {p}")
