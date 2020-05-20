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


class Bernoulli:
    def __init__(self):
        self._phix = None
        self._phiy = 0

    def train(self, input, classes):

        phiy = sum(classes) / len(classes)

        pos = sum(classes)
        neg = len(classes) - pos

        m, n = input.shape

        p1Vec = np.zeros(n)
        p0Vec = np.zeros(n)

        for i in range(n):
            p0Num = 1
            p1Num = 1
            p0Den = 2 + neg
            p1Den = 2 + pos

            for j in range(m):
                if classes[j] == 1:
                    p1Num += input[j, i]
                else:
                    p0Num += input[j, i]

            p1Vec[i] = np.log(p1Num / p1Den)
            p0Vec[i] = np.log(p0Num / p0Den)

        # p0Num = np.ones(input.shape[1])
        # p1Num = np.ones(input.shape[1])
        #
        # p0Den = 2.
        # p1Den = 2.
        #
        # for i in range(len(classes)):
        #     if classes[i] == 1:
        #         p1Num += input[i]
        #         p1Den += sum(input[i])
        #     else:
        #         p0Num += input[i]
        #         p0Den += sum(input[i])
        # p1Vec = np.log(p1Num / p1Den)
        # p0Vec = np.log(p0Num / p0Den)

        phix = np.array([p0Vec, p1Vec])

        self._phix = phix
        self._phiy = phiy
        return phix, phiy

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

    def createVocList(self, dataSet):
        vocSet = set([])
        for word in dataSet:
            vocSet = vocSet | set(word)
        return np.array(vocSet)

    def word2vec(self, input, vocList):
        retVec = [0] * len(vocList)
        for word in input:
            if word in vocList:
                retVec[vocList.index(word)] = 1
            else:
                print('\nthe word {} is not in voc.\n'.format(word))
        return retVec
