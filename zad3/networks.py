#!/usr/bin/env python3

import torch
from torch import nn

class NeuralNetwork1(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.CNNstack = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels=5, kernel_size=(3,3), padding = 1, padding_mode='zeros'),
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels = 5, out_channels = 10, kernel_size = (3,3), padding = 1, padding_mode = 'zeros')
        )

        self.LinearStack = nn.Sequential(
            nn.Linear(14*14*10, 150),
            nn.ReLU(),
            nn.Linear(150,10)
        )

    def forward(self, x):
        logits = self.CNNstack(x)
        logits = logits.view(-1, 14*14*10)
        logits = self.LinearStack(logits)

        return logits


class NeuralNetwork2(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.CNNstack = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 3, kernel_size=(5,5), padding = 2, padding_mode = 'zeros'), #28x28x3
            nn.MaxPool2d(2,2), #14x14x3
            nn.Conv2d(in_channels = 3, out_channels = 8,kernel_size = (3,3), padding = 1, padding_mode = 'zeros'), #14x14x8
            nn.Conv2d(in_channels = 8, out_channels = 1, kernel_size = (1,1)), #14x14x1
            nn.Conv2d(in_channels = 1, out_channels = 3, kernel_size = (3,3), padding = 1, padding_mode = 'zeros') #14x14x3
        )
        
        self.LinearStack = nn.Sequential(
            nn.Linear(14*14*3, 150),
            nn.ReLU(),
            nn.Linear(150,10)
        )

    def forward(self, x):
        logits = self.CNNstack(x)
        logits = logits.view(-1, 14*14*3)
        logits = self.LinearStack(logits)

        return logits


class NeuralNetwork3(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.CNNstack = nn.Sequential(#28x28x1
            nn.Conv2d(in_channels = 1, out_channels = 4, kernel_size=(5,5)), #24x24x4
            nn.Conv2d(in_channels = 4, out_channels = 8, kernel_size=(5,5)), #20x20x8
            nn.Conv2d(in_channels = 8, out_channels = 12, kernel_size=(3,3)), #18x18x12
            nn.Conv2d(in_channels = 12, out_channels = 1, kernel_size = (1,1)) #18x18x1
        )
        
        self.LinearStack = nn.Sequential(
            nn.Linear(18*18*1, 200),
            nn.ReLU(),
            nn.Linear(200, 80),
            nn.ReLU(),
            nn.Linear(80, 10)
        )

    def forward(self, x):
        logits = self.CNNstack(x)
        logits = logits.view(-1, 18*18*1)
        logits = self.LinearStack(logits)

        return logits
