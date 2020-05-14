import numpy as np
import matplotlib.pyplot as plt

from Main.ClassificationFunction.Weights import Weights


def get_theta(x, y, W=None):
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


class LR:
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
            theta = get_theta(X, y, W)
            preds = X.dot(theta)
            self._theta = theta
        else:
            '''Then LWR'''
            for idx, X_eval in enumerate(X):
                w = W(X_eval, X, k)
                theta = get_theta(X, y, w)
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
            theta = get_theta(X, y, w)
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
