import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchstain
from torchvision import transforms
from PIL import Image


class PCamDataset(Dataset):
    """
    A simple custom Dataset for PCam data stored in .h5 files.
    Assumes data is of shape (N, H, W, C) and labels are shape (N,).
    """
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]  # shape (H, W, C) as NumPy array
        label = int(self.labels[idx])

        if self.transform:
            img = self.transform(img)

        return img, label
    
    
class Macenko(nn.Module):
    def __init__(self, reference_image, target_W=None, alpha=1, beta=0.01):
        super(Macenko, self).__init__()
        self.target_W = target_W
        self.alpha = alpha
        self.beta = beta
        self.reference_image = reference_image.astype(np.uint8)
    
    def forward(self, image):
        """
        Apply Macenko normalization to a single image with error handling.
        
        Parameters:
            image (np.ndarray): The image to normalize, shape (H, W, C) in RGB format.
            reference_image (np.ndarray): The reference image for normalization, shape (H, W, C) in RGB format.
        
        Returns:
            np.ndarray: The normalized image, shape (C, H, W) in normalized format.
            None: If normalization fails for any reason.
        """
        try:
            # # Set up the transformation
            # T = transforms.Compose([
            #     transforms.ToTensor(),
            # ])

            # Initialize the MacenkoNormalizer
            normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')

            # Fit the normalizer with the reference image
            normalizer.fit(self.reference_image)

            # Transform the image and apply normalization
            t_to_transform = image
            norm_img, _, _ = normalizer.normalize(I=t_to_transform, stains=True)

            # Return the normalized image
            return norm_img

        except torch.linalg.LinAlgError as e:
            # print(f"LinAlgError during normalization: {e}")
            pass
        except Exception as e:
            # print(f"Unexpected error during normalization: {e}")
            pass

        # Return None if normalization fails
        return image


def get_data_loaders(
    train_data_path="./data/pcam/training_split.h5",
    val_data_path="./data/pcam/validation_split.h5",
    test_data_path="./data/pcam/test_split.h5",
    train_label_path="./data/pcam/Labels/Labels/camelyonpatch_level_2_split_train_y.h5",
    val_label_path="./data/pcam/Labels/Labels/camelyonpatch_level_2_split_valid_y.h5",
    test_label_path="./data/pcam/Labels/Labels/camelyonpatch_level_2_split_test_y.h5",
    batch_size=128,
    num_workers=12,
    transform = None
):
    # Loading image data
    with h5py.File(train_data_path, 'r') as f:
        train_data = f['x'][:]
    with h5py.File(val_data_path, 'r') as f:
        val_data = f['x'][:]
    with h5py.File(test_data_path, 'r') as f:
        test_data = f['x'][:]

    # Loading labels
    with h5py.File(train_label_path, 'r') as f:
        train_labels = f['y'][:].reshape(-1,)
    with h5py.File(val_label_path, 'r') as f:
        val_labels = f['y'][:].reshape(-1,)
    with h5py.File(test_label_path, 'r') as f:
        test_labels = f['y'][:].reshape(-1,)

    # Build datasets
    train_dataset = PCamDataset(train_data, train_labels, transform=transform)
    val_dataset   = PCamDataset(val_data, val_labels, transform=transform)
    test_dataset  = PCamDataset(test_data, test_labels, transform=transform)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
