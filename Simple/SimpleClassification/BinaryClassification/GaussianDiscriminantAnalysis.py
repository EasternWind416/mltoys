import numpy as np


def gaussian(phi, mu, sigma, x):
    pred = np.zeros((x.shape[0], 1))
    for idx, x_eval in enumerate(x):
        mean1 = x_eval - mu[1]
        mean0 = x_eval - mu[0]
        p1 = phi * np.exp(-0.5 * mean1.dot(np.linalg.inv(sigma)).dot(mean1.T))
        p0 = (1 - phi) * np.exp(-0.5 * mean0.dot(np.linalg.inv(sigma)).dot(mean0.T))
        if p1 >= p0:
            pred[idx] = 1
        else:
            pred[idx] = 0
        return pred


def accuracy(pred, y):
    return np.sum([pred == y]) / y.shape[0]


class GDA():

    def __init__(self, x, y):
        self._x = x
        self._y = y

        self._phi = None
        self._mu= None
        self._sigma = None

    def fit(self):
        x = self._x
        y = self._y

        m = len(y)
        m1 = np.sum(y)
        m0 = m - m1

        phi = m1 / m

        mu0 = np.sum(x[y==0], axis=0) / m0
        mu1 = np.sum(x[y==1], axis=0) / m1
        mu = np.stack([mu0, mu1], axis=0)

        sigma = np.zeros((x.shape[1], x.shape[1]))
        for i in range(m):
            sig = np.matrix(x[i] - mu[int(y[i])])
            sigma += sig.T.dot(sig)
        sigma /= m

        pred = gaussian(phi, mu, sigma, x)
        acc = accuracy(pred, y)

        print('After GDA fitting, acc = ', acc)
        print('GDA params: \n\tphi = {0}, mu0 = {1}, mu1 = {2}, sigma = {3}'.format(phi, mu0, mu1, sigma))

        self._phi = phi
        self._mu = mu
        self._sigma = sigma

        return phi, mu, sigma

    def prediction(self, x):
        return gaussian(self._phi, self._mu, self._sigma, x)
