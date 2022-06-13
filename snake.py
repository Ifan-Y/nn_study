import matplotlib as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import matplotlib.pyplot as plt


class Universe:
    def __init__(self):
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        self.a = [1, 1]
        self.first = 0
        self.second = 1
        plt.ion()
        # plt.show()

    def run(self):
        self.a[self.first] += 1
        print(self.a)
        a = self.a
        plt.gca().add_patch(plt.Rectangle((a[0], a[1]), 2, 2))


if __name__ == '__main__':
    n = Universe()
    n.run()
