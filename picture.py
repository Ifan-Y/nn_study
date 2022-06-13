import matplotlib.pyplot as plt
import math
import numpy as np
import torch


def sigmoid(w):
    [rows, lines] = w.shape
    o = np.zeros(shape=(rows, lines))
    for i in range(rows):
        for j in range(lines):
            o[i, j] = 1 / (1 + pow(math.e, -w[i, j]))
    return o


def weighting(input_s, w_s):
    lens = len(w_s)
    input_step = input_s
    for i in range(lens):
        output_step = w_s[i] * input_step
        o_step = sigmoid(output_step)
        input_step = o_step
    return input_step


input = np.mat([0.9, 0.1, 0.8]).T
w1 = np.mat([[0.9, 0.3, 0.4], [0.2, 0.8, 0.2], [0.1, 0.5, 0.6]])
w2 = np.mat([[0.3, 0.7, 0.5], [0.6, 0.5, 0.2], [0.8, 0.1, 0.9]])
w = [w1, w2]

output = weighting(input, w)
print(output)
# out1 = w1 * input
# o1 = sigmoid(out1)
# output = w2 * o1
# outputing = sigmoid(output)
# print(o1)
# # print(out1)
# print(output)
# print(outputing)
# for x in np.arange(-5, 5, 0.001):
#     a.append(x)
#     y.append(1 / (1 + pow(math.e, -x)))
#     i += 1
#
# plt.plot(a, y)
# plt.show()
