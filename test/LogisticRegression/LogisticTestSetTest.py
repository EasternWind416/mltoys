import numpy as np
import matplotlib.pyplot as plt

from Main.Logistic.LogisticRegression import LR


numFeat = len(open('./data/TestSet.txt', 'r').readline().strip().split('\t')) - 1
with open('./data/TestSet.txt', 'r') as file:
    x = []
    y = []
    for line in file.readlines():
        curX = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            curX.append(float(curLine[i]))
        y.append(float(curLine[-1]))
        x.append(curX)
    x = np.array(x)
    y = np.array(y)

X = x
Y = y

regression = LR(X, Y, lr=.01, DescentMethod='SGD', iterNum=500, activation='sigmoid')
theta, pred = regression.fit()
x1 = [max(x[:, 0]), min(x[:, 0])]
x2 = [-(theta[0] + theta[1] * i) / theta[2] for i in x1]

x_class_1 = x[y == 1]
x_class_0 = x[y == 0]
plt.plot(x_class_1[:, 0], x_class_1[:, 1], '.', label='class 1')
plt.plot(x_class_0[:, 0], x_class_0[:, 1], '.', label='class 0')
plt.plot(x1, x2)
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()
