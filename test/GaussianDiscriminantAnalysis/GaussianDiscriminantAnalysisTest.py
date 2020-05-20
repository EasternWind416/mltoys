import numpy as np

from ml.GDA import GDA

x_path = './LegoTest/logistic_x.txt'
y_path = './LegoTest/logistic_y.txt'

with open(y_path, 'r') as yfile:
    ystr = yfile.read()
    y = ystr.split()
    length = len(y)
    for i in range(len(y)):
        y[i] = float(y[i])

y = np.array(y).reshape((-1,))
y[y < 1] = 0

with open(x_path, 'r') as xfile:
    xstr = xfile.read()
    x = xstr.split()
    x1 = []
    x2 = []
    for i in range(len(x)):
        if i % 2 == 0:
            x1.append(float(x[i]))
        else:
            x2.append(float(x[i]))

x1 = np.array(x1).reshape((-1,))
x2 = np.array(x2).reshape((-1,))
x = np.stack([x1, x2],axis=1)

gda = GDA(x, y)
phi, mu, sigma = gda.fit()
