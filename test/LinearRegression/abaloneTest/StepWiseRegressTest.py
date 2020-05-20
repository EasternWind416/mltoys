import numpy as np
import matplotlib.pyplot as plt

from ml.Regress import StepWiseRegress


def loadData(fileName):
    numFeat = len(open(fileName, 'r').readline().strip().split('\t')) - 1
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

fileName = './data/abalone.txt'
x, y = loadData(fileName)

yMean = y.mean()
y = y - yMean

xMean = x.mean(axis=0)
xVar = x.var(axis=0)
x = (x - xMean) / xVar

returnMat, *_ = StepWiseRegress().regress(x, y, 0.01, 200)

plt.plot(returnMat)
plt.show()
