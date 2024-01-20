#!/usr/bin/env python3

import torch 
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = datasets.FashionMNIST(
        root = "data",
        train = True,
        download = False,
        transform = ToTensor()
)

test_data = datasets.FashionMNIST(
        root = "data",
        train = False,
        download = False,
        transform = ToTensor()
)

device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequantial(
            
            #tu jakoś trzeba dodać te splotowe

        )


    def forward(self, x):

        #może trzeba zrobić dwa sequentiale
        #najpierw odpalić splotowy a potem połączyć
        #odpalić flatten i dopiero gęste


        #nie chcemy tego chyba robić 
        #chyba że warstwy splotowe mają nie być w lin_rel_stack
        x = self.flatten(x)

        logits = self.linear_relu_stack(x)
        return logits









