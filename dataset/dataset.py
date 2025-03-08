from typing import Tuple

import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from configs import Config


class CINIC10(Dataset):
    """CINIC10 dataset contants."""
    cinic_directory = 'data/CINIC-10'
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]


def get_loader(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    cinic_train = torchvision.datasets.ImageFolder(CINIC10.cinic_directory + '/train',
    	transform=transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=CINIC10.cinic_mean,std=CINIC10.cinic_std)]))
    cinic_valid = torchvision.datasets.ImageFolder(CINIC10.cinic_directory + '/valid',
    	transform=transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=CINIC10.cinic_mean,std=CINIC10.cinic_std)]))

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
