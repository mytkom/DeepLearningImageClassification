import os
from typing import Tuple

import numpy as np
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import v2
from torch.utils.data import default_collate

from configs import Config

class CINIC10(Dataset):
    """CINIC10 dataset contants."""

    cinic_directory = "CINIC-10"
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]


def get_loader(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    augmentation = v2.Identity()  # No augmentation

    if cfg.data.augmentation == "BasicTransform":
        augmentation = v2.Compose(
            [
                v2.RandomCrop(32, padding=4),
                v2.RandomRotation(15),
                v2.RandomAdjustSharpness(sharpness_factor=2),
            ]
        )
    elif cfg.data.augmentation == "BasicColors":
        augmentation = v2.Compose(
            [
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]
        )
    elif cfg.data.augmentation == "AutoAugment":
        augmentation = v2.AutoAugment()
    elif cfg.data.augmentation == "All":
        augmentation = v2.Compose(
            [
                v2.RandomCrop(32, padding=4),
                v2.RandomRotation(15),
                v2.RandomAdjustSharpness(sharpness_factor=2),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                v2.AutoAugment(),
            ]
        )

    collate = default_collate
    if cfg.data.mix_augmentations:
        cutmix = v2.CutMix(num_classes=cfg.data.num_classes)
        mixup = v2.MixUp(num_classes=cfg.data.num_classes)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
        collate = lambda batch: cutmix_or_mixup(*default_collate(batch))

    train_transform = v2.Compose(
        [
            augmentation,
            v2.ToTensor(),
            v2.Normalize(mean=CINIC10.cinic_mean, std=CINIC10.cinic_std),
        ]
    )

    cinic_train = torchvision.datasets.ImageFolder(
        os.path.join(cfg.data.root, CINIC10.cinic_directory, "train"),
        transform=train_transform
    )
    cinic_valid = torchvision.datasets.ImageFolder(
        os.path.join(cfg.data.root, CINIC10.cinic_directory, "valid"),
        transform=v2.Compose(
            [
                v2.ToTensor(),
                v2.Normalize(mean=CINIC10.cinic_mean, std=CINIC10.cinic_std),
            ]
        ),
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
        collate_fn=collate,
        shuffle=True,
    )
    val_loader = DataLoader(
        cinic_valid,
        num_workers=cfg.evaluation.num_workers,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
    )
    return train_loader, val_loader
