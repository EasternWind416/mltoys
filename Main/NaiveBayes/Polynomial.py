import numpy as np


class Polynomial:
    def __init__(self):
        self._phix = None
        self._phiy = 0

    def train(self, input, label):
        p0Num = np.ones(input.shape[1])
        p1Num = np.ones(input.shape[1])
        p0Den = 2
        p1Den = 2

        for i in range(input.shape[0]):
            if label[i] == 1:
                p1Num += input[i]
                p1Den += sum(input[i])
            else:
                p0Num += input[i]
                p0Den += sum(input[i])

        p0 = np.log(p0Num / p0Den)
        p1 = np.log(p1Num / p1Den)

        self._phix = np.array([p0, p1])
        self._phiy = sum(label) / len(label)
        return self._phix, self._phiy

    def predict(self, input):
        phix = self._phix
        phiy = self._phiy

        if len(input.shape) < 2:
            m = 1
            for i in range(m):
                p1 = sum(input * phix[1]) + np.log(phiy)
                p0 = sum(input * phix[0]) + np.log(1. - phiy)

                if p1 > p0:
                    return 1
                else:
                    return 0
        else:
            m, *_ = input.shape
            pred = np.zeros(m)

            for i in range(m):

                p1 = sum(input[i] * phix[1]) + np.log(phiy)
                p0 = sum(input[i] * phix[0]) + np.log(1. - phiy)

                if p1 > p0:
                    pred[i] = 1
                else:
                    pred[i] = 0

            return pred
