import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import pandas as pd
import os
import shutil
from torch import nn
from torch import optim

data_list = pd.read_csv('mnist_chinese/chinese_mnist.csv')


def make_train_dir():
    for i in range(15):
        os.makedirs(f'data/test/test/{i}')
    for j in range(15):
        os.makedirs(f'data/test/train/{j}')

    for i in range(15000):
        print(i)
        single_data = data_list.iloc[i]
        # print(single_data[2])
        file_name = f"input_{single_data[0]}_{single_data[1]}_{single_data[2]}.jpg"
        # print(file_name)
        shutil.copy(f'mnist_chinese/data/{file_name}', f'data/test/test/{single_data[3]}')
        print(f"{file_name} done\n")
        # print(single_data[3])


def make_test_dir():
    test_img_dirs = 'data/test/test'
    train_img_dirs = 'data/test/train'
    lists = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000, 10000, 100000000]
    for i in lists:
        test_img_dir = f"{test_img_dirs}/{i}"
        train_img_dir = f"{train_img_dirs}/{i}"
        os.makedirs(train_img_dir)
        test_img_files = os.listdir(test_img_dir)
        j = 0
        for k in test_img_files:
            test_img_file = f"{test_img_dir}/{k}"
            shutil.move(test_img_file, train_img_dir)
            j += 1
            if j >= 150:
                break


def load_data():
    transform = transforms.Compose([
        # transforms.Resize((28, 28)),
        transforms.Grayscale(1),
        transforms.ToTensor(),
    ])
    dataset_test = datasets.ImageFolder('data/test/test', transform=transform)
    dataset_train = datasets.ImageFolder('data/test/train', transform=transform)

    tran_data_size = len(dataset_train)
    # print(f"number_1={tran_data_size}")
    test_data_size = len(dataset_test)
    # print(f"number_2={test_data_size}")

    dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
    dataloader_test = DataLoader(dataset_test, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

    return dataloader_train, dataloader_test


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
            nn.Linear(12000, 3000),
            nn.ReLU(),
            nn.Linear(3000, 750),
            nn.ReLU(),
            nn.Linear(750, 188),
            nn.ReLU(),
            nn.Linear(188, 15),
            nn.LogSoftmax(),
        )
        # self.network_nn = nn.Sequential(
        #     nn.Linear(64*64, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 15),
        # )

        # self.network_nn = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=1,
        #         out_channels=6,
        #         kernel_size=5,
        #         stride=1,
        #         padding=2
        #     ),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        # )
    def forward(self, x):
        in_size = x.size(0)
        out = self.network_1st(x)
        out = out.view(in_size, -1)
        out = self.network_2nd(out)
        x = torch.flatten(x, 1)
        # out = self.network_nn(x)
        return out


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")

model = Net().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

train_dataloader, test_dataloader = load_data()


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
            print(f"loss: {loss} [{current}/{size}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, current = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            current += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    current /= size
    print(f"Test Error: \n Accuracy: {(100 * current):>0.1f}%, Avg loss: {test_loss:>8f}\n")


epochs = 10
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model, "chinese__mnist.pt")
print("Saved Pytorch Model State to chinese_mnist.pt")
