import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize, Pad, RandomCrop, RandomHorizontalFlip, RandomErasing
from cifar import MY_CIFAR10,MY_CIFAR100
from svhn import MY_SVHN
from fmnist import MY_FMNIST
from kmnist import MY_KMNIST

np.random.seed(2)

def cifar10_dataloaders(data_dir,rate):
    print('Data Preparation')
    cifar10_train_ds = MY_CIFAR10(data_dir, train=True, download=True,rate_partial=rate)
    train_loader = torch.utils.data.DataLoader(
        cifar10_train_ds,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    print('Loading dataset {0} for training -- Num_samples: {1}, num_classes: {2}'.format(datasets.CIFAR10.__name__,len(cifar10_train_ds),10))

    test_transform = Compose([
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # Data loader for test dataset
    cifar10_test_ds = datasets.CIFAR10(data_dir, transform=test_transform, train=False, download=True)
    print('Test set -- Num_samples: {0}'.format(len(cifar10_test_ds)))
    test = DataLoader(
        cifar10_test_ds, batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    return train_loader, test


def svhn_dataloaders(data_dir,rate):
    print('Data Preparation')    
    svhn_train_ds = MY_SVHN(data_dir, split='train', download=True,rate_partial=rate)
    train_loader = torch.utils.data.DataLoader(
        svhn_train_ds,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    print('Loading dataset {0} for training -- Num_samples: {1}, num_classes: {2}'.format(datasets.SVHN.__name__,len(svhn_train_ds),10))

    test_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    # Data loader for test dataset
    svhn_test_ds = datasets.SVHN(data_dir, transform=test_transform, split='test', download=True)
    print('Test set -- Num_samples: {0}'.format(len(svhn_test_ds)))
    test = DataLoader(
        svhn_test_ds, batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    return train_loader, test



def cifar100_dataloaders(data_dir,rate):
    print('Data Preparation')
    cifar100_train_ds = MY_CIFAR100(data_dir, train=True, download=True,rate_partial=rate)
    train_loader = torch.utils.data.DataLoader(
        cifar100_train_ds,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    print('Loading dataset {0} for training -- Num_samples: {1}, num_classes: {2}'.format(datasets.CIFAR100.__name__,len(cifar100_train_ds),100))

    test_transform = Compose([
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # Data loader for test dataset
    cifar100_test_ds = datasets.CIFAR100(data_dir, transform=test_transform, train=False, download=True)
    print('Test set -- Num_samples: {0}'.format(len(cifar100_test_ds)))
    test = DataLoader(
        cifar100_test_ds,
		batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    return train_loader, test


def fmnist_dataloaders(data_dir,rate):
    print('Data Preparation')
    fmnist_train_ds = MY_FMNIST(data_dir, train=True, download=True, rate_partial=rate)
    train_loader = torch.utils.data.DataLoader(
        fmnist_train_ds,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    print('Loading dataset {0} for training -- Num_samples: {1}, num_classes: {2}'.format(datasets.FashionMNIST.__name__,len(fmnist_train_ds), 10))
    test_transform = Compose([
        ToTensor(),
        Normalize((0.1307), (0.3081)),
    ])
    # Data loader for test dataset
    fmnist_test_ds = datasets.FashionMNIST(data_dir, transform=test_transform, train=False, download=True)
    print('Test set -- Num_samples: {0}'.format(len(fmnist_test_ds)))
    test = DataLoader(
        fmnist_test_ds,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    return train_loader, test


def kmnist_dataloaders(data_dir,rate):
    print('Data Preparation')
    kmnist_train_ds = MY_KMNIST(data_dir, train=True, download=True, rate_partial=rate)
    train_loader = torch.utils.data.DataLoader(
        kmnist_train_ds,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    print('Loading dataset {0} for training -- Num_samples: {1}, num_classes: {2}'.format(datasets.KMNIST.__name__, len(kmnist_train_ds), 10))
    test_transform = Compose([
        ToTensor(),
        Normalize((0.5), (0.5)),
    ])
    # Data loader for test dataset
    kmnist_test_ds = datasets.KMNIST(data_dir, transform=test_transform, train=False, download=True)
    print('Test set -- Num_samples: {0}'.format(len(kmnist_test_ds)))
    test = DataLoader(
        kmnist_test_ds,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    return train_loader, test