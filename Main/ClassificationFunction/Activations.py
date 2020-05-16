import numpy as np


class Activation():

    def getActivation(self, activation):

        def sigmoid(z):
            return 1. / (1 + np.exp(-z))
        
        def tanh(z):
            return 2 * 1. / (1 + np.exp(-2 * z)) - 1

        switch = {'sigmoid': sigmoid,
                  'tanh': tanh}
        return switch.get(activation)