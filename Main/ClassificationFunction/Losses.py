import numpy as np


class Loss():

    def getLossFunction(self, loss):

        def MSE(preds, y):
            return np.mean((preds - y) ** 2)

        switch = {'MSE': MSE}
        return switch.get(loss)