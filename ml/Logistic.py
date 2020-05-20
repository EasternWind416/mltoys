import numpy as np

from ml.ClassificationFunction.Activations import Activation
from ml.ClassificationFunction.Losses import Loss
from ml.ClassificationFunction.GradientDescentMethod import GDMethod

class LR():

    '''
    Now is the normal logistic regression, whose threshold is 0.5.
    TO DO:
        Construct this class to generalization which threshold can change with some situation.
    '''
    def __init__(self, x, y,
                 lr=0.1,
                 es=1e-6,
                 iterNum=100,
                 DescentMethod='BGD',
                 activation='sigmoid',
                 loss='MSE'):
        '''

        :param x: with shape (N, M-1), which N is the # of train LegoTest and M is the degree;
        :param y: with shape (N,),which should be 0 or 1;
        :param lr: learning rate;
        :param es: terminating error;
        :param iterNum: terminating epochs;
        :param loss: to evaluate error.
        '''
        x0 = np.ones((x.shape[0],1))
        self._x = np.concatenate([x0, x], axis=1)
        self._y = y
        self._theta = np.ones(self._x.shape[1])
        self._pred = None

        self._lr = lr
        self._es = es
        self._iterNum = iterNum

        if isinstance(activation, str):
            self._g = Activation().getActivation(activation)
        else:
            self._g = activation

        if isinstance(loss, str):
            self._lossFunction = Loss().getLossFunction(loss)
        else:
            self._lossFunction = loss

        if isinstance(DescentMethod, str):
            self._GDMethod = GDMethod().getGDMethod(DescentMethod)
        else:
            self._GDMethod = DescentMethod

    def fit(self):

        x = self._x
        y = self._y
        theta = self._theta
        lr = self._lr

        theta = self._GDMethod(x, y, theta, lr,
                               self._g, self._lossFunction,
                               self._iterNum, self._es)

        pred = x.dot(theta)

        self._theta = theta
        self._pred = pred

        return self._theta, self._pred

    def prediction(self, x):

        x0 = np.ones(x.shape[0]).reshape(-1, 1)
        X = np.concatenate([x0, x], axis=1)

        pred = X.dot(self._theta)

        return pred
