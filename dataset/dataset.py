import os
from typing import Tuple, Optional, List, Union, Dict
import numpy as np
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset, Sampler
from torchvision.transforms import v2
from torch.utils.data import default_collate
from configs import Config


class CINIC10(Dataset):
    """CINIC10 dataset contants."""
    cinic_directory = "CINIC-10"
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]

class TaskSampler(Sampler):
    def __init__(self, dataset: Dataset, n_way: int, n_shot: int, n_query: int, n_tasks: int):
        super().__init__(data_source=None)
        self.dataset = dataset
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_tasks = n_tasks
        if isinstance(dataset, Subset):
            subset_indices = dataset.indices
            imgs = [dataset.dataset.imgs[i] for i in subset_indices]
        else:
            imgs = dataset.imgs

        self.labels = [instance[1] for instance in imgs]

        self.items_per_label: Dict[int, List[int]] = {}
        for item, label in enumerate(self.labels):
            if label in self.items_per_label:
                self.items_per_label[label].append(item)
            else:
                self.items_per_label[label] = [item]

    def __iter__(self):
        for _ in range(self.n_tasks):
            selected_classes = np.random.choice(list(self.items_per_label.keys()),
                                                self.n_way, replace=False)

            batch_indices = []
            batch_labels = []

            for class_idx, cls in enumerate(selected_classes):
                support_indices = np.random.choice(
                    self.items_per_label[cls],
                    self.n_shot,
                    replace=False
                )
                batch_indices.extend(support_indices.tolist())
                batch_labels.extend([class_idx] * self.n_shot)

            for class_idx, cls in enumerate(selected_classes):
                remaining_indices = [
                    i for i in self.items_per_label[cls]
                    if i not in batch_indices[:self.n_way * self.n_shot]
                ]
                query_indices = np.random.choice(
                    remaining_indices,
                    self.n_query,
                    replace=False
                )
                batch_indices.extend(query_indices.tolist())
                batch_labels.extend([class_idx] * self.n_query)

            yield batch_indices

    def __len__(self):
        return self.n_tasks

def get_standard_loader(cfg: Config, custom_transforms: Optional[v2.Compose] = None) -> Tuple[DataLoader, DataLoader]:
    augmentation = v2.Identity()  # No augmentation

    if custom_transforms:
        base_transforms = custom_transforms
    else:
        base_transforms = v2.Compose(
            [
                v2.ToTensor(),
                v2.Normalize(mean=CINIC10.cinic_mean, std=CINIC10.cinic_std)
            ]
        )

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
            base_transforms
        ]
    )

    val_transform = base_transforms

    cinic_train = torchvision.datasets.ImageFolder(
        os.path.join(cfg.data.root, CINIC10.cinic_directory, "train"),
        transform=train_transform
    )
    cinic_valid = torchvision.datasets.ImageFolder(
        os.path.join(cfg.data.root, CINIC10.cinic_directory, "valid"),
        transform=val_transform,
    )

    if cfg.data.subset_size:
        def balanced_subset_indices(dataset, subset_size):
            class_indices = {}
            for idx, (_, label) in enumerate(dataset.imgs):
                if label not in class_indices:
                    class_indices[label] = []
                class_indices[label].append(idx)

            per_class_size = subset_size // len(class_indices)
            subset_indices = []
            for indices in class_indices.values():
                subset_indices.extend(np.random.choice(indices, per_class_size, replace=False))

            return subset_indices

        train_indices = balanced_subset_indices(cinic_train, cfg.data.subset_size)
        val_indices = balanced_subset_indices(cinic_valid, cfg.data.subset_size)

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


def get_few_shot_loader(cfg: Config, custom_transforms: Optional[v2.Compose] = None) -> Tuple[DataLoader, DataLoader]:
    if custom_transforms:
        base_transforms = custom_transforms
    else:
        base_transforms = v2.Compose(
            [
                v2.ToTensor(),
                v2.Normalize(mean=CINIC10.cinic_mean, std=CINIC10.cinic_std)
            ]
        )

    cinic_train = torchvision.datasets.ImageFolder(
        os.path.join(cfg.data.root, CINIC10.cinic_directory, "train"),
        transform=base_transforms
    )
    cinic_valid = torchvision.datasets.ImageFolder(
        os.path.join(cfg.data.root, CINIC10.cinic_directory, "valid"),
        transform=base_transforms,
    )

    if cfg.data.subset_size:
        def balanced_subset_indices(dataset, subset_size):
            class_indices = {}
            for idx, (_, label) in enumerate(dataset.imgs):
                if label not in class_indices:
                    class_indices[label] = []
                class_indices[label].append(idx)

            per_class_size = subset_size // len(class_indices)
            subset_indices = []
            for indices in class_indices.values():
                subset_indices.extend(np.random.choice(indices, per_class_size, replace=False))

            return subset_indices

        train_indices = balanced_subset_indices(cinic_train, cfg.data.subset_size)
        val_indices = balanced_subset_indices(cinic_valid, cfg.data.subset_size)

        cinic_train = Subset(cinic_train, train_indices)
        cinic_valid = Subset(cinic_valid, val_indices)

    train_sampler = TaskSampler(
        cinic_train, n_way=cfg.few_shot.n_way, n_shot=cfg.few_shot.n_shot, n_query=cfg.few_shot.n_query,
        n_tasks=cfg.few_shot.n_training_episodes
    )
    val_sampler = TaskSampler(
        cinic_valid, n_way=cfg.few_shot.n_way, n_shot=cfg.few_shot.n_shot, n_query=cfg.few_shot.n_query,
        n_tasks=cfg.few_shot.n_validation_tasks
    )

    train_loader = DataLoader(
        cinic_train,
        num_workers=cfg.training.num_workers,
        batch_sampler=train_sampler,
    )
    val_loader = DataLoader(
        cinic_valid,
        num_workers=cfg.evaluation.num_workers,
        batch_sampler=val_sampler,
        shuffle=False,
    )
    return train_loader, val_loader


def get_loader(cfg: Config, custom_transforms: Optional[v2.Compose] = None) -> Tuple[DataLoader, DataLoader]:
    if cfg.model.is_prototypical:
        return get_few_shot_loader(cfg, custom_transforms)
    return get_standard_loader(cfg, custom_transforms)
