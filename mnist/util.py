import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import condensa.data

def mnist_train_val_loader(train_batch_size, val_batch_size, root='./data', random_seed=42, shuffle=True):
    """
    Splits the MNIST training set into training and validation
    sets (9:1 split) and returns the corresponding data loaders.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.MNIST(root=root,
                                          train=True,
                                          download=True,
                                          transform=transform_train)
    valset = torchvision.datasets.MNIST(root=root,
                                        train=True,
                                        download=True,
                                        transform=transforms.ToTensor())

    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(0.1 * num_train)  # 10% for validation

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]
    trainsampler = SubsetRandomSampler(train_idx)
    valsampler = SubsetRandomSampler(val_idx)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=train_batch_size,
                                              sampler=trainsampler,
                                              num_workers=8)
    valloader = torch.utils.data.DataLoader(valset,
                                            batch_size=val_batch_size,
                                            sampler=valsampler,
                                            num_workers=8)

    return trainloader, valloader


def mnist_test_loader(batch_size, root='./data'):
    """
    Construct an MNIST test dataset loader.
    """
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    testset = torchvision.datasets.MNIST(root=root,
                                         train=False,
                                         download=True,
                                         transform=transform_test)
    
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=8)

    return testloader


# util.py
import torch

def count_params(model):
    """
    Count the number of trainable parameters in the given PyTorch model.

    Args:
    - model (torch.nn.Module): The PyTorch model.

    Returns:
    - int: The number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_device():
    """
    Set the device to CUDA if available, otherwise CPU.

    Returns:
    - torch.device: The device (CUDA or CPU).
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

