#!/usr/bin/env python3

import torch 
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from networks import NeuralNetwork1, NeuralNetwork2, NeuralNetwork3


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


model = NeuralNetwork1().to(device)
model2 = NeuralNetwork2().to(device)
model3 = NeuralNetwork3().to(device)
#model.load_state_dict(torch.load('model.p'))
print(model)
print(model2)
print(model3)

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
        optimizer.step()
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








