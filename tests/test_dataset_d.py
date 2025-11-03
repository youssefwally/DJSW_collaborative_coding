from pathlib import Path
import numpy as np
import torch
from utils.dataset_mnist03_h5 import Mnist03Dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed"

def test_d_data_available():
    assert Path(DATA_DIR / "mnist03.h5").exists(), "No data available. Run download_prepare_mnist03.py to make the MNIST03 dataset available."

def test_mnist03_dataset():
    dataset = Mnist03Dataset(DATA_DIR / "mnist03.h5", split='train', normalize=True, flatten=True)
    X,y = dataset[0]
    if X.ndim > 2:
        assert X.shape == (1,28,28), "Shape for MNIST03 images should be (1,28,28)."
    else:
        X.shape == (784,), "Shape for flattened MNIST03 images should be (784,)."

    assert X.dtype == torch.float32, "Type of MNIST03 images should be float32"
    assert isinstance(X, torch.Tensor), "MNIST03 image should be a torch Tensor."
    y_int = int(y.item())  # get scalar value
    assert 0 <= y_int <= 3, "Label y can only be between 0 and 3."
    assert isinstance(y, torch.Tensor), "MNIST03 label should be a torch Tensor."
    assert y.dtype == torch.long, "Label tensor should have dtype torch.long."
    assert len(dataset) > 0, "The dataset is empty."

