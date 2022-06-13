# coding: utf-8

# import torchdata.datapipes as dp
# import numpy as np
# from torchvision.io import read_image
# import pandas as pd
# from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
# import matplotlib.pyplot as plt
import torch
from torch import nn

training_data = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break  # Get cpu or gpu device for training.

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device.")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(28 * 28, 196),
            # nn.Sigmoid(),
            # nn.ReLU(),
            # nn.Linear(392, 196),
            # # # nn.Sigmoid(),
            # nn.ReLU(),
            # nn.Linear(196, 98),
            # nn.ReLU(),
            # nn.Linear(98, 49),
            nn.ReLU(),
            nn.Linear(196, 10),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        result = self.linear_sigmoid_stack(x)
        return result


model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    model.train()
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss:{loss} [{current}/{size}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, current = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            # print(f"y:{y}")
            # print(f"Result{pred}")
            test_loss += loss_fn(pred, y).item()
            current += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    current /= size
    print(f"Test Error: \n Accuracy: {(100 * current):>0.1f}%, Avg loss: {test_loss:>8f}\n")


epochs = 1

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print("Done!")

torch.save(model, "myself.pth")
print("Saved Pytorch Model to myself.pt")

# --------------------------

input_shape = 28 * 28  # 输入数据,改成自己的输入shape

# #set the model to inference mode
model.eval()

x = torch.randn(batch_size, input_shape)  # 生成张量
x = x.to(device)
export_onnx_file = "myself.onnx"  # 目的ONNX文件名
torch.onnx.export(model,
                  x,
                  export_onnx_file,
                  opset_version=10,
                  do_constant_folding=True,  # 是否执行常量折叠优化
                  input_names=["input"],  # 输入名
                  output_names=["output"],  # 输出名
                  dynamic_axes={"input": {0: "batch_size"},  # 批处理变量
                                "output": {0: "batch_size"}})
