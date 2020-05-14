import numpy as np

from Simple.SimpleClassification.ClassificationFunction.Activations import Activation
from Simple.SimpleClassification.ClassificationFunction.Loss import Loss


def rec(pred):

    mask_pos = pred >= 0.5
    mask_neg = pred < 0.5

    pred[mask_pos] = 1
    pred[mask_neg] = 0

    return pred


def accuracy(pred, y):

    pred = rec(pred)

    mask_acc = y == pred
    acc = np.sum(mask_acc) / y.shape[0]
    return acc


class LR():

    '''
    Now is the normal logistic regression, whose threshold is 0.5.
    TO DO:
        Construct this class to generalization which threshold can change with some situation.
    '''
    def __init__(self, x, y, lr=0.1, es=1e-6, epochs=100, loss='MSE'):
        '''

        :param x: with shape (N, M-1), which N is the # of train data and M is the degree;
        :param y: with shape (N,),which should be 0 or 1;
        :param lr: learning rate;
        :param es: terminating error;
        :param epochs: terminating epochs;
        :param loss: to evaluate error.
        '''
        x0 = np.ones((x.shape[0],1))
        self._x = np.concatenate([x0, x], axis=1)
        self._y = y
        self._theta = np.zeros(self._x.shape[1])
        self._pred = None
        self._acc = 0

        self._lr = lr
        self._es = es
        self._epochs = epochs

        self._activation = Activation('sigmoid')
        self._g = self._activation.getActivation()

        self._lossClass = Loss(loss)
        self._lossFunction = self._lossClass.getLossFunction()

    def fit(self):

        x = self._x
        y = self._y
        theta = self._theta
        pred = x.dot(theta)

        e = float('inf')
        epoch = 0
        acc = 0

        while e > self._es:
            # and epoch < self._epochs:

            epoch += 1
            print('\n---epoch {}---'.format(epoch))

            z = x.dot(theta)
            g = self._g(z)
            e_t = g - y
            grade = x.T.dot(e_t)
            theta_old = theta.copy()
            theta -= self._lr * grade
            pred = x.dot(theta)

            e = np.sum(abs(theta - theta_old))
            print('now theta shift: {}'.format(e))

            loss = self._lossFunction(pred, y)
            print('now loss: {}'.format(loss))

            acc = accuracy(pred, y)
            print('now acc: {}'.format(acc))

        self._theta = theta
        self._pred = pred
        self._acc = acc

        return self._theta, self._pred, self._acc

    def prediction(self, x):

        x0 = np.ones(x.shape[0])
        X = np.stack([x0, x], axis=1)

        pred = rec(X.dot(self._theta))

        return pred
