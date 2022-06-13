import numpy
import math
import random
import matplotlib as mpt
import matplotlib.pyplot as plt

a = [1.5, 1.5]
b = [2, 1]

A = random.uniform(-100.0, 100.0)
print(A)
x_a = [0, 3]
y_a = [0, 3 * A]
plt.plot(x_a, y_a, color='black')

# y = A * a[0]

b = -100.0
c = 100.0

while 1.5 * A >= 1.5 or 2 * A <= 1:
    if 1.5 * A >= 1.5:
        c = A
        A = random.uniform(b, A)
        x_a = [0, 3]
        y_a = [0, 3 * A]
        plt.plot(x_a, y_a, color='y')
        print(A)
    else:
        b = A
        A = random.uniform(A, 1)
        x_a = [0, 3]
        y_a = [0, 3 * A]
        plt.plot(x_a, y_a, color='g')
        print(A)

x_a = [0, 3]
y_a = [0, 3 * A]
plt.plot([1.5, 2], [1.5, 1], 'o')
plt.plot(x_a, y_a, color='r')
plt.show()
