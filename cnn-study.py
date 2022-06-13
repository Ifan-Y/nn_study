# coding: utf-8

import torch
from torch.autograd import Variable
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# load datas
train_dataset = datasets.MNIST('data/', download=True, train=True,
                               transform=transforms.Compose([transforms.ToTensor(),
                                                             transforms.Normalize((0.1307,), (0.3081,)), ]),
                               )
test_dataset = datasets.MNIST('data/', download=True, train=True,
                              transform=transforms.Compose([transforms.ToTensor(),
                                                            transforms.Normalize((0.1307,), (0.3081,)), ]),
                              )

# 数据迭代器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.network_1st = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 120, kernel_size=5),
            nn.ReLU(),
        )
        self.network_2nd = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.LogSoftmax(),
        )

    def forward(self, x):
        in_size = x.size(0)
        out = self.network_1st(x)
        out = out.view(in_size, -1)
        out = self.network_2nd(out)
        return out


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")

model = Net().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item, batch * len(X)
            print(f"loss: {loss}  [{current}/{size}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_baches = len(dataloader)
    model.eval()
    test_loss, current = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            current += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_baches
    current /= size
    print(f"Test Error: \n Accuracy: {(100 * current):>0.1f}%, Avg loss: {test_loss:>8f}\n")


epochs = 10
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)
print("Done!")

torch.save(model, "cnn_1.pt")
print("Saved Pytorch Model State to cnn_1.pt")
