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

train_loader = DataLoader(train_data, batch_size=50, shuffle=True)
test_loader = DataLoader(test_data, batch_size=50, shuffle=True)


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
        self.CNNstack = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=5 ,kernel_size=(3,3),padding=1, padding_mode='zeros'),

        nn.MaxPool2d(2,2),
        nn.Conv2d(in_channels=5, out_channels = 10, kernel_size=(3,3), padding=1, padding_mode='zeros')
        )

        self.Linear = nn.Sequential(
            nn.Linear(14*14*10,150),
            nn.ReLU(),
            nn.Linear(150,10)
        )


    def forward(self, x):
        logits = self.CNNstack(x)
        logits = logits.view(-1, 14*14*10)
        logits = self.Linear(logits)

        return logits


model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)

def train(model, train_loader, loss_fn, device):
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        train_loss += loss.item()
    return train_loss / len(train_loader)


#walidacja
def validation(model, test_loader, loss_fn, device):
    val_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return val_loss / len(test_loader), accuracy

best_val_loss = float('inf')
for epoch in range(10):
    train_loss = train(model, train_loader, loss_fn, device)
    val_loss, accuracy = validation(model, test_loader, loss_fn, device)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience = 0
        torch.save(model.state_dict(), 'model.p')


    else:
        patience += 1


    if patience > 3:
        print("Early stopping reached")
        break

    print(f"Epoch: {epoch:02} - Train loss: {train_loss:02.3f} - Validation loss: {val_loss:02.3f} - Accuracy: {accuracy:02.3f}")









