import numpy as np

class Loss():

    def __init__(self, loss):
        self._lossFunction = loss

    def getLossFunction(self):

        def MSE(preds, y):
            return np.mean((preds - y) ** 2)

        switch = {'MSE': MSE}
        return switch.get(self._lossFunction)