import numpy as np
import matplotlib.pyplot as plt

from ml.Regress import RidgeRegress


def laodData(fileName):
    numFeat = len(open(fileName, 'r').readline().split('\t')) - 1
    dataMat = []
    labelMat = []

    with open(fileName, 'r') as file:
        for line in file.readlines():
            lineArr = []
            curLine = line.strip().split('\t')
            for i in range(numFeat):
                lineArr.append(float(curLine[i]))
            dataMat.append(lineArr)
            labelMat.append(float(curLine[-1]))
    return np.array(dataMat), np.array(labelMat)


x, y = laodData('./data/abalone.txt')

# 预处理
yMean = y.mean()
y = y - yMean
xMean = x.mean(axis=0)
xVar = x.var(axis=0)
x = (x - xMean) / xVar

numTest = 30

w = np.zeros((numTest, x.shape[1] + 1))

for i in range(numTest):
    theta, *_ = RidgeRegress().regress(x, y, np.exp(i - 10))
    w[i, :] = theta.T

xz = [(i - 10) for i in range(30)]

plt.plot(xz, w)
plt.show()
