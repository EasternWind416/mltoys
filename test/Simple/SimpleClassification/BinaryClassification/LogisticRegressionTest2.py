import numpy as np
import matplotlib.pyplot as plt

from Simple.SimpleClassification.BinaryClassification.LogisticRegression import LR

path = './data/data_b.txt'

with open(path, 'r') as xfile:
    y = []
    x1 = []
    x2 = []
    xstr = xfile.read()
    x = xstr.split()
    for i in range(len(x)):
        if i % 3 == 0:
            y.append(float(x[i]))
        elif i % 3 == 1:
            x1.append(float(x[i]))
        else:
            x2.append(float(x[i]))

y = np.array(y).reshape((-1,))
y[y < 1] = 0
x1 = np.array(x1).reshape((-1,))
x2 = np.array(x2).reshape((-1,))
x = np.stack([x1, x2], axis=1)

LR = LR(x, y, lr=0.1)
theta, pred, acc = LR.fit()

print('\nacc: {}'.format(acc))

_x1 = np.array([np.max(x1), np.min(x1)])
_x2 = (theta[0] + theta[1] * _x1) / (-theta[2])

plot = np.stack([y, x1, x2], axis=1)

neg = []
pos = []

for i in range(len(y)):
    if plot[i, 0] < 1:
        neg.append(plot[i])
    else:
        pos.append(plot[i])

neg_len = len(neg)

plot[:neg_len] = np.array(neg)
plot[neg_len:] = np.array(pos)

plt.plot(plot[:neg_len, 1], plot[:neg_len, 2], '.', label='neg')
plt.plot(plot[neg_len:, 1], plot[neg_len:, 2], '.', label='pos')
plt.plot(_x1, _x2, label='classification', color='red')
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()