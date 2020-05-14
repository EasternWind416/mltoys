"""
岭回归算法其实是在线性回归的基础上加入了正则化。
可以更好的理解数据，忽视无关项。
"""
import numpy as np
import matplotlib.pyplot as plt

from Main.ClassificationFunction.Weights import Weights


def get_theta(x, y, lamda=0.2):
    I = np.ones(x.shape[1])
    I = np.diag(I)
    return np.linalg.inv(x.T.dot(x) + lamda * I).dot(x.T).dot(y)


class RR:
    def __init__(self):
        self._preds = None
        self._theta = None
        self._x = None
        self._y = None

    def regress(self, x, y, lamda=0.2):

        X0 = np.ones((x.shape[0], 1))
        if len(x.shape) < 2:
            x = x.reshape(-1, 1)
        X = np.concatenate([X0, x], axis=1)

        theta = get_theta(X, y, lamda)
        preds = X.dot(theta)

        self._x = X
        self._y = y
        self._preds = preds
        self._theta = theta

        return theta, preds

    def plot(self, label=None):

        x = self._x
        y = self._y
        preds = self._preds

        plt.plot(x, y, '.', label='origin')
        plt.plot(x, preds, label='ridge regression')

        if label is None:
            label = ['x', 'y']
        else:
            label = label
        plt.xlabel(label[0])
        plt.ylabel(label[1])
        plt.show()
