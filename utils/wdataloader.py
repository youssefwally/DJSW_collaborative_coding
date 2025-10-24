import torch
import h5py
import hdf5plugin
import numpy as np
from torch.utils.data import Dataset

class USPS06Dataset(Dataset):
    """
    Dataset class for filtered USPS data (digits 0-6) stored in H5 file.
    """
    def __init__(self, h5_file='../data/raw/usps.h5', set_type="train", transform=None):
        """
        Args:
            h5_file: Path to the H5 file containing the data
            set_type: Which dataset to load ('train', 'val', or 'test')
            transform: Optional transform to be applied on samples
        """
        self.h5_file = h5_file
        self.set_type = set_type
        self.transform = transform

        # Load and filter data to include only labels 0-6
        with h5py.File(self.h5_file, 'r') as f:
            train_X = f['train_data'][:]
            train_y = f['train_labels'][:]
            val_X = f['val_data'][:]
            val_y = f['val_labels'][:]
            test_X = f['test_data'][:]
            test_y = f['test_labels'][:]

            mask = train_y <= 6
            train_data = train_X[mask]
            train_labels = train_y[mask]
            mask = val_y <= 6
            val_data = val_X[mask]
            val_labels = val_y[mask]
            mask = test_y <= 6
            test_data = test_X[mask]
            test_labels = test_y[mask]

        # Save filtered data back to new H5 file
        with h5py.File('../data/processed/usps.h5', 'w') as new_file:
            new_file.create_dataset('train_data', data=train_data)
            new_file.create_dataset('train_labels', data=train_labels)
            new_file.create_dataset('val_data', data=val_data)
            new_file.create_dataset('val_labels', data=val_labels)
            new_file.create_dataset('test_data', data=test_data)
            new_file.create_dataset('test_labels', data=test_labels)

        self.h5_file = self.h5_file.replace('raw', 'processed')

        # Get length without loading all data
        with h5py.File(self.h5_file, 'r') as f:
            if set_type== "train":
                self.length = len(f['train_labels'])
                self.data_key = 'train_data'
                self.labels_key = 'train_labels'
            elif set_type== "val":
                self.length = len(f['val_labels'])
                self.data_key = 'val_data'
                self.labels_key = 'val_labels'
            else:
                self.length = len(f['test_labels'])
                self.data_key = 'test_data'
                self.labels_key = 'test_labels'
    
    def __len__(self):
        return self.length
    
    def get_input_dim(self):
        """
        Returns the input dimension of the dataset samples.
        """
        with h5py.File(self.h5_file, 'r') as f:
            _, c, w, h = f[self.data_key].shape
            return c*w*h
    
    def __getitem__(self, idx):
        # Load only the requested sample from disk
        with h5py.File(self.h5_file, 'r') as f:
            image = torch.from_numpy(f[self.data_key][idx].flatten()).float()
            label = torch.from_numpy(np.array(f[self.labels_key][idx])).long()
        
        if self.transform:
            image = self.transform(image)
        
        return image, label