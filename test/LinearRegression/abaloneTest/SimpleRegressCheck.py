"""
检验各种简单回归方法在未知数据上的准确性。
"""
import numpy as np

from Main.Regression.LinearRegress import LR


def loadDataSet(fileName):
    """
    加载数据
        解析以tab键分隔的文件中的浮点数
    Returns：
        dataMat ：  feature 对应的数据集
        labelMat ： feature 对应的分类标签，即类别标签

    """
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return np.array(dataMat), np.array(labelMat)


def rssError(preds, y):
    return ((preds - y) ** 2).sum()


fileName = './data/abalone.txt'
x, y = loadDataSet(fileName)
w = 'Gaussian'

# 使用不同平滑系数的高斯函数在训练集上进行对比
old01 = LR(x[: 99], y[: 99], w).LWR(0.201)
old1 = LR(x[: 99], y[: 99], w).LWR(1)
old10 = LR(x[: 99], y[: 99], w).LWR(10)

print('old 01 error size is: ', rssError(old01, y[: 99]))
print('old 1 error size is: ', rssError(old1, y[: 99]))
print('old 10 error size is: ', rssError(old10, y[: 99]))

# 使用不同平滑系数的高斯函数在测试集上进行对比
new01 = LR(x[: 99], y[: 99], w).LWR_test(x[100: 199], x[: 99], 0.234)
new1 = LR(x[: 99], y[: 99], w).LWR_test(x[100: 199], x[: 99], 1)
new10 = LR(x[: 99], y[: 99], w).LWR_test(x[100: 199], x[: 99], 10)

print('new 01 error size is: ', rssError(new01, y[100: 199]))
print('new 1 error size is: ', rssError(new1, y[100: 199]))
print('new 10 error size is: ', rssError(new10, y[100: 199]))

# 使用简单线性回归进行比较
lr = LR(x[: 99], y[: 99])
lr.LWR()
theta = lr.getTheta()
newx = x[100: 199]
ones = np.ones(newx.shape[0]).reshape(-1,1)
X = np.concatenate([ones, newx], axis=1)
preds = X.dot(theta)
print('linear regression error size is: ', rssError(preds, y[100: 199]))