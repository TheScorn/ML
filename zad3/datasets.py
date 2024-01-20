#!/usr/bin/env python3


import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.pyplot as plt


#loadery danych
train_data = datasets.FashionMNIST(
        root="data", #gdzie są dane
        train=True, #czy pobrać treningowy czy testowy
        download=True,
        transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
)
