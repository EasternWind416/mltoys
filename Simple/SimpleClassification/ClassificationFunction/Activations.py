import numpy as np

class Activation():

    def __init__(self, activation):
        self._activation = activation

    def getActivation(self):

        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        switch = {'sigmoid': sigmoid}
        return switch.get(self._activation)