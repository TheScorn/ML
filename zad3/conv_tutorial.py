#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as otpim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
from torchvision import transforms, datasets
import torch.nn.functional as F
torch.manual_seed(42)


from PIL import Image
from dataclasses import dataclass
import matplotlib.pyplot as plt
import random
import numpy as np

#tworzymy własną klasę datasetu
class FashionMNIST(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.dataset = datasets.ImageFolder(self.data_dir, transform = self.transform)

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            return self.dataset[index]
#te 3 metody trzeba zdefiniować żeby klasa działała


#możemy użyć pytorcha do tworzenia transformacji danych, by je uogólnić
#za pomocą Compose można stworzyć pakiet transformacji żeby nie musieć ich wpisywać za każdym razem
data_transforms = transforms.Compose([
    transforms.Resize((28,28)),
    #można wpisać z jakim prawdopodobieństwem będą działać transformacje
    transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandomRotation(10),
    #małe zdjęcia to może za mocno zaburzyć informacje
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


#nie chce działaś, zły typ danych do wczytania w ten sposób chyba
#dataset = FashionMNIST('data/FashionMNIST/raw/', transform = data_transforms)




