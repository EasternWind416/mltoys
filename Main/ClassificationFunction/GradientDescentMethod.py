import numpy as np


class GDMethod:

    def getGDMethod(self, method):

        def BGD(x, y,
                theta, lr,
                gfunc, lossfunc,
                iterNum, es):

            e = float('inf')
            iter = 0

            while e > es and iter < iterNum:
                iter += 1
                print('\n---iter {}---'.format(iter))

                z = x.dot(theta)
                g = gfunc(z)
                e_t = g - y
                grade = x.T.dot(e_t)
                theta_old = theta.copy()
                theta -= lr * grade
                pred = x.dot(theta)

                e = np.sum(abs(theta - theta_old))
                print('now theta shift: {}'.format(e))

                loss = lossfunc(pred, y)
                print('now loss: {}'.format(loss))
            return theta

        def SGD_demo(x, y,
                     theta, lr,
                     gfunc, lossfunc,
                     iterNum, es):

            e = float('inf')
            iter = 0

            m, n = x.shape

            while e > es and iter < iterNum:
                for i in range(m):
                    iter += 1
                    print('\n---iter {}---'.format(iter))

                    z = x[i].dot(theta)
                    g = gfunc(z)
                    e_t = g - y[i]
                    grade = x[i] * e_t
                    theta_old = theta.copy()
                    theta -= lr * grade
                    pred = x.dot(theta)

                    e = np.sum(abs(theta - theta_old))
                    print('now theta shift: {}'.format(e))

                    loss = lossfunc(pred, y)
                    print('now loss: {}'.format(loss))
            return theta

        def SGD(x, y,
                theta, lr,
                gfunc, lossfunc,
                iterNum, es):

            e = float('inf')
            iter = 0

            m, n = x.shape

            dataIndx = range(m)

            while iter < iterNum:
                iter += 1
                for i in range(m):
                    alpha = 4 / (1. + iter + i) + lr / 100.
                    randomIdx = int(np.random.uniform(0, len(dataIndx)))
                    print('---iter {}, random {}---'.format(iter, i))

                    z = x[dataIndx[randomIdx]].dot(theta)
                    g = gfunc(z)
                    e_t = g - y[dataIndx[randomIdx]]
                    grade = x[dataIndx[randomIdx]] * e_t

                    theta -= alpha * grade
                    pred = x.dot(theta)

                    loss = lossfunc(pred, y)
                    print('now loss: ', loss)
            return theta

        switch = {'BGD': BGD,
                  'SGD_demo': SGD_demo,
                  'SGD': SGD}
        return switch.get(method)
