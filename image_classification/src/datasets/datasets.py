import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass 
from torchvision import datasets, transforms
import numpy as np

def numpy_collate(batch):
        if isinstance(batch[0], np.ndarray):
            return np.stack(batch)
        elif isinstance(batch[0], (tuple,list)):
            transposed = zip(*batch)
            return [numpy_collate(samples) for samples in transposed]
        else:
            return np.array(batch)

@dataclass
class MNISTData:
    train_loader = DataLoader
    test_loader = DataLoader
    batch_size: int

    def __post_init__(self):
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('../data', train=False, transform=transform)
        print("Length train dataset: ", len(train_dataset))
        print("Length test dataset: ", len(test_dataset))

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=numpy_collate)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=numpy_collate)
