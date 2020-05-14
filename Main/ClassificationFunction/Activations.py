import numpy as np


class Activation():

    def getActivation(self, activation):

        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        switch = {'sigmoid': sigmoid}
        return switch.get(activation)