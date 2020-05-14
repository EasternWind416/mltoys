"""
逐步向前算法是一种贪心算法。
初试化权重W为0，然后每次增佳或减少一个很小的值。
步骤：
    数据标准化；
    迭代：
        设置最小误差lErr为正无穷；
        对每个特征：
            增大或缩小：
                改变一个系数得到权重W；
                计算当前W下的新误差Err；
                如果Err小于lErr，设置wBest = w， lErr = Err
            w = wBest
"""
import numpy as np


def ErrLoss(preds, origin):
    return ((preds - origin) ** 2).sum()


class SW:
    def regress(self, x, y, step=0.1, numIter=100):
        num, feat = x.shape
        w = np.zeros((feat, 1))
        wTemp = w.copy()
        wBest = w.copy()

        for i in range(numIter):
            lErr = np.inf
            for j in range(feat):
                for sign in [-1, 1]:
                    wTemp = wBest.copy()
                    wTemp[j] += step * sign
                    yTemp = x.dot(wTemp)
                    err = ErrLoss(yTemp, y)
                    if err < lErr:
                        lErr = err
                        wBest = wTemp
            w = wBest.copy()
            preds = x.dot(w)
        return w, preds