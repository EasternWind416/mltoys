import numpy as np
import matplotlib.pyplot as plt

from Main.LinearRegression import LR


path = './data/data.txt'

with open(path, 'r') as file:
    y = []
    x1 = []
    x2 = []
    str = file.read()
    data = str.split()
    for i in range(len(data)):
        if i % 3 == 0:
            y.append(float(data[i]))
        if i % 3 == 1:
            x1.append(float(data[i]))
        if i % 3 == 2:
            x2.append(float(data[i]))
    y = np.array(y)
    x1 = np.array(x1)
    x2 = np.array(x2)

idx = x1.argsort()

y = x2[idx]
x = x1[idx]

lr = LR(x, y)
preds = lr.LWR()

lr.plot(['x1', 'x2'])
