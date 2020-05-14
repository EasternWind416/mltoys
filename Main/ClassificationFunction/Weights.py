import numpy as np
from math import exp

class Weights:

    def getWeightFunction(self, weight):

        def Gaussian(x_i, x, k=0.01):
            w = np.zeros((x.shape[0], x.shape[0]))
            for i in range(x.shape[0]):
                diff = x_i - x[i, :]
                w[i, i] = exp(diff.dot(diff.T) / (-2 * k ** 2))
            return w

        switch = {'Gaussian': Gaussian}
        return switch.get(weight)
