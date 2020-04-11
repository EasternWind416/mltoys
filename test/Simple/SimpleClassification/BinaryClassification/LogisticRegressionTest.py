import numpy as np
import matplotlib.pyplot as plt

from Simple.SimpleClassification.BinaryClassification.LogisticRegression import LogisticRegression

x_path = './data/logistic_x.txt'
y_path = './data/logistic_y.txt'

with open(y_path, 'r') as yfile:
    ystr = yfile.read()
    y = ystr.split()
    length = len(y)
    for i in range(len(y)):
        y[i] = float(y[i])

y = np.array(y).reshape((length,))
y[y < 1] = 0

with open(x_path, 'r') as xfile:
    x1 = []
    x2 = []
    xstr = xfile.read()
    x = xstr.split()
    for i in range(len(x)):
        if i % 2 == 0:
            x1.append(float(x[i]))
        else:
            x2.append(float(x[i]))

x1 = np.array(x1).reshape((length,))
x2 = np.array(x2).reshape((length,))
x = np.stack([x1, x2], axis=1)

LR = LogisticRegression(x, y,lr=0.01)
theta, pred, acc = LR.fit()

print('\nacc: {}'.format(acc))

_x1 = np.array([np.max(x1), np.min(x1)])
_x2 = (theta[0] + theta[1] * _x1) / (-theta[2])

plt.plot(x1[:50], x2[:50], '.', label='neg')
plt.plot(x1[50:], x2[50:], '.', label='pos')
plt.plot(_x1, _x2, label='classification', color='red')
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()