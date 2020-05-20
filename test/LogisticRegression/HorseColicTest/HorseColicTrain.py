import numpy as np

from ml.Logistic import LR


def loadData(fileName):
    numFeat = len(open(fileName, 'r').readline().strip().split('\t')) - 1
    with open(fileName, 'r') as file:
        x = []
        y = []
        for line in file.readlines():
            curLine = line.strip().split('\t')
            curX = []
            for i in range(numFeat):
                curX.append(float(curLine[i]))
            x.append(curX)
            y.append(float(curLine[-1]))
        x = np.array(x)
        y = np.array(y)
    return x, y

fileName = './data/HorseColicTraining.txt'
fileName_test = './data/HorseColicTest.txt'


def rec(pred):
    pred[pred >= .5] = 1
    pred[pred < .5] = 0
    return pred


def acc(pred, y):
    flag = pred == y
    return np.sum(flag) / len(y)


x, y = loadData(fileName)
x_test, y_test = loadData(fileName_test)

regression = LR(x, y, lr=0.1, iterNum=500, DescentMethod='SGD')
theta, pred = regression.fit()

ac = acc(rec(pred), y)
print('\ntraining acc: ', ac)

pred = regression.prediction(x_test)
ac = acc(rec(pred), y_test)
print('\ntest acc: ', ac)