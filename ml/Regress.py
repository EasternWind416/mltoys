import numpy as np
import matplotlib.pyplot as plt

from ml.ClassificationFunction.Weights import Weights


class LinearRegress:
    def __init__(self, x, y, W=None):
        self._x = x
        self._y = y

        if isinstance(W, str):
            self._W = Weights().getWeightFunction(W)
        else:
            self._W = W

        self._theta = None
        self._preds = None

    def LWR(self, k=0.01):
        """

        :param x: is a tensor with shape (N, M-1), which is lack of interception;
        :param y: is the ground truth with shape (N,);
        :param W: is a function to construct weighted matrix, which should have format like W(x, x_eval);
        :param k: is the smooth params;
        :return: prediction.
        """

        x = self._x
        y = self._y
        W = self._W

        X0 = np.ones((x.shape[0], 1))
        if len(x.shape) < 2:
            x = x.reshape(-1, 1)
        X = np.concatenate([X0, x], axis=1)
        preds = []
        Theta = []

        if W is None:
            '''Then LWR is the normal LR'''
            theta = self.get_theta(X, y, W)
            preds = X.dot(theta)
            self._theta = theta
        else:
            '''Then LWR'''
            for idx, X_eval in enumerate(X):
                w = W(X_eval, X, k)
                theta = self.get_theta(X, y, w)
                Theta.append(theta)
                pred = X_eval.dot(theta)
                preds.append(pred)
            preds = np.array(preds)
            Theta = np.array(Theta)
            self._theta = Theta

        self._preds = preds
        return preds

    def LWR_test(self, x_test, x_origin, k=0.01):
        """
        test version only for lwr.
        :param x_test: x in test;
        :param x_origin: x in origin;
        :param k: smooth param;
        :return: preds;
        """

        x_test = x_test
        x = x_origin
        y = self._y
        W = self._W

        X0 = np.ones((x.shape[0], 1))
        if len(x.shape) < 2:
            x = x.reshape(-1, 1)
        X = np.concatenate([X0, x], axis=1)

        X0 = np.ones((x_test.shape[0], 1))
        if len(x_test.shape) < 2:
            x_test = x_test.reshape(-1, 1)
        X_test = np.concatenate([X0, x_test], axis=1)

        preds = []

        for idx, X_eval in enumerate(X_test):
            w = W(X_eval, X, k)
            theta = self.get_theta(X, y, w)
            pred = X_eval.dot(theta)
            preds.append(pred)
        preds = np.array(preds)

        return preds

    def getTheta(self):
        return self._theta

    def plot(self, label=None):

        x = self._x
        y = self._y
        preds = self._preds

        plt.plot(x, y, '.', label='origin')
        plt.plot(x, preds, label='linear regression')

        if label is None:
            label = ['x', 'y']
        else:
            label = label
        plt.xlabel(label[0])
        plt.ylabel(label[1])
        plt.show()

    def get_theta(self, x, y, W=None):
        """

        :param x: is a tensor with shape (N, M), which N is the # of LegoTest, M is the degrees of polynomial;
        :param y: is the ground truth with shape (N,);
        :param W: is the weighted matrix;
        :return: parameter theta.
        """
        if W is None:
            return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
        else:
            return np.linalg.inv(x.T.dot(W).dot(x)).dot(x.T).dot(W).dot(y)


class RidgeRegress:
    """
    岭回归算法其实是在线性回归的基础上加入了正则化。
    可以更好的理解数据，忽视无关项。
    """
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

        theta = self.get_theta(X, y, lamda)
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

    def get_theta(self, x, y, lamda=0.2):
        I = np.ones(x.shape[1])
        I = np.diag(I)
        return np.linalg.inv(x.T.dot(x) + lamda * I).dot(x.T).dot(y)


class StepWiseRegress:
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
    def regress(self, x, y, step=0.1, numIter=100):
        num, feat = x.shape
        w = np.zeros((feat, 1))
        returnMat = np.zeros((numIter, feat))
        wTemp = w.copy()
        wBest = w.copy()

        for i in range(numIter):
            print('iter times: ', i)
            lErr = np.inf
            for j in range(feat):
                for sign in [-1, 1]:
                    wTemp = wBest.copy()
                    wTemp[j] += step * sign
                    yTemp = x.dot(wTemp)
                    err = self.ErrLoss(yTemp, y)
                    if err < lErr:
                        lErr = err
                        wBest = wTemp
            w = wBest.copy()
            returnMat[i, :] = w.T
            preds = x.dot(w)
        return returnMat, w, preds

    def ErrLoss(self, preds, origin):
        return ((preds - origin) ** 2).sum()
