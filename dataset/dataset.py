import os
from typing import Tuple, Optional

import numpy as np
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import transforms
from torchvision.transforms.autoaugment import AutoAugment

from configs import Config


class CINIC10(Dataset):
    """CINIC10 dataset contants."""

    cinic_directory = "CINIC-10"
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]


def get_loader(cfg: Config, custom_transforms: Optional[transforms.Compose] = None) -> Tuple[DataLoader, DataLoader]:
    if custom_transforms:
        base_transforms = custom_transforms
    else:
        base_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=CINIC10.cinic_mean, std=CINIC10.cinic_std)
            ]
        )

    if cfg.data.augmentation == "BasicTransform":
        augmentation = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomRotation(15),
                transforms.RandomAdjustSharpness(sharpness_factor=2),
            ]
        )
    elif cfg.data.augmentation == "BasicColors":
        augmentation = transforms.Compose(
            [
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]
        )
    elif cfg.data.augmentation == "AutoAugment":
        augmentation = AutoAugment()
    else:
        augmentation = transforms.Compose([])  # No augmentation

    train_transform = transforms.Compose(
        [
            augmentation,
            base_transforms
        ]
    )

    val_transform = base_transforms

    cinic_train = torchvision.datasets.ImageFolder(
        os.path.join(cfg.data.root, CINIC10.cinic_directory, "train"),
        transform=train_transform,
    )
    cinic_valid = torchvision.datasets.ImageFolder(
        os.path.join(cfg.data.root, CINIC10.cinic_directory, "valid"),
        transform=val_transform,
    )

    if cfg.data.subset_size:
        train_indices = np.random.choice(
            len(cinic_train), cfg.data.subset_size, replace=False
        )
        val_indices = np.random.choice(
            len(cinic_valid), cfg.data.subset_size, replace=False
        )
        cinic_train = Subset(cinic_train, train_indices)
        cinic_valid = Subset(cinic_valid, val_indices)

    train_loader = DataLoader(
        cinic_train,
        num_workers=cfg.training.num_workers,
        batch_size=cfg.training.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        cinic_valid,
        num_workers=cfg.evaluation.num_workers,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
    )
    return train_loader, val_loader
