import numpy as np


class svm_smo_nk_simple:
    """
    svm smo algorithm simple version with non kernel.
    """

    def __init__(self):
        self._x = None
        self._y = None
        self._alpha = None
        self._b = None

    def get_j(self, i, m):
        """

        :param i: alpha的下标i
        :param m: 总的数量
        :return: alpha的下标j
        """
        j = i
        while j == i:
            j = int(np.random.uniform(0, m))
        return j

    def rec_alpha(self, a, l, h):
        """

        :param a: 需要整流的值
        :param L: 最小值
        :param H: 最大值
        :return:
        """
        if a > h:
            a = h
        if a < l:
            a = l
        return a

    def run(self, x, y, c, tol, max_iter_num):
        """

        :param x: input
        :param y: labels
        :param c: 松弛变量
        :param tol: 容错率
        :param max_iter_num: 最大迭代数量
        :return: b, 模型常量
                 alpha, 乘子
        """
        x = np.mat(x)
        y = np.mat(y).reshape((-1, 1))
        m, n = x.shape

        b = 0
        alpha = np.mat(np.zeros((m, 1)))

        iter = 0
        while (iter < max_iter_num):
            '''
            查看alpha是否优化过
            '''
            alphaPairChanged = 0
            for i in range(m):
                predi = float(np.multiply(alpha, y).T * (x * x[i, :].T)) + b
                ei = predi - y[i]

                '''
                误差大于容错率才考虑优化：
                    abs(y[i] * ei)超出tol.
                
                还需检验样本(xi, yi)是否满足KKT条件：
                    yi * predi >= 1, alpha = 0     ===> 在边界两侧
                    yi * predi == 1, 0 < alpha < C ===> 在边界上
                    yi * predi <= 1, alpha == c    ===> 在边界中间
                '''
                if ((y[i] * ei < - tol) and (alpha[i] < c)) \
                        or ((y[i] * ei > tol) and (alpha[i] > 0)):

                    '''
                    如果i点可以进行优化，则随机选取j点。
                    '''
                    j = self.get_j(i, m)
                    predj = float(np.multiply(alpha, y).T * (x * x[j, :].T)) + b
                    ej = predj - y[j]

                    old_alpha_i = alpha[i].copy()
                    old_alpha_j = alpha[j].copy()

                    '''
                    使用rec_alpha函数的L和H，
                    将alpha[j]调整到0~C之间，
                    如果L=H, 则continue跳出循环。
                    y[i] != y[j]表示两点处于异侧，需要相减；否则同侧，需要相加。
                    见：
                        https://blog.csdn.net/crazy_programmer_p/article/details/38553757
                    '''
                    if y[i] != y[j]:
                        l = max(0, alpha[j] - alpha[i])
                        h = min(c, alpha[j] - alpha[i] + c)
                    else:
                        l = max(0, alpha[j] + alpha[i] - c)
                        h = min(c, alpha[j] + alpha[i])
                    '''如果l=h，就不用优化了'''
                    if l == h:
                        print('L == H')
                        continue

                    '''
                    计算eta = k11 + k22 - 2 * k12
                    eta > 0.
                    '''
                    eta = x[i, :] * x[i, :].T + x[j, :] * x[j, :].T - 2. * x[i, :] * x[j, :].T
                    if eta <= 0:
                        print('eta <= 0')
                        continue

                    alpha[j] += y[j] * (ei - ej) / eta
                    alpha[j] = self.rec_alpha(alpha[j], l, h)

                    '''
                    检查alpha j改变幅度是否显著，如果不显著，就推出当前循环
                    '''
                    if abs(alpha[j] - old_alpha_j) < 1e-5:
                        print('alpha j dit not moving enough')
                        continue

                    '''
                    alpha i 改变方向应该和 j 相反
                    但是大小一样，因为要保证和不变。
                    '''
                    alpha[i] += y[j] * y[i] * (old_alpha_j - alpha[j])

                    '''
                    b = y - wx:
                        bi = f(xi) - yi * ai * kii - yj * aj * kij - Σyl * al * kil
                    '''
                    b1 = b - ei - \
                         y[i] * (alpha[i] - old_alpha_i) * x[i, :] * x[i, :].T - \
                         y[j] * (alpha[j] - old_alpha_j) * x[i, :] * x[j, :].T
                    b2 = b - ej - \
                         y[i] * (alpha[i] - old_alpha_i) * x[i, :] * x[j, :].T - \
                         y[j] * (alpha[j] - old_alpha_j) * x[j, :] * x[j, :].T

                    if (alpha[i] > 0) and (alpha[i] < c):
                        b = b1
                    elif (alpha[j] > 0) and (alpha[j] < c):
                        b = b2
                    else:
                        b = (b1 + b2) / 2.
                    alphaPairChanged += 1
                    print('iter: {}, i: {}, pairs changed: {}'.format(iter, i, alphaPairChanged))
            if alphaPairChanged == 0:
                iter += 1
            else:
                iter = 0
            print('iteration number: ', iter)
        self._x = x
        self._y = y
        self._alpha = alpha
        self._b = b
        return b, alpha

    def cal_w(self):
        x = self._x
        y = self._y
        alpha = self._alpha
        b = self._b
        m, n = x.shape
        w = np.zeros((n, 1))
        for i in range(m):
            w += np.multiply(alpha[i] * y[i], x[i, :].T)
        return w


class svm_smo_nonkernel:

    def __init__(self, input, label):
        self._input = np.mat(input)
        self._label = np.mat(label).reshape((-1, 1))
        self._Ecache = np.mat(np.zeros((self._input.shape[0], 2)))

        self._alpha = np.mat(np.zeros((self._input.shape[0], 1)))
        self._b = 0

    def run(self, max_iter, c, tol):

        cur_iter = 0
        pair_changed = 0
        entire_set = True

        while (cur_iter < max_iter and pair_changed > 0) or entire_set:
            pair_changed = 0

            if entire_set:
                for i in range(self._input.shape[0]):
                    pair_changed += self._innerLoop(i, c, tol)
                    print('full set, iter {} i: {}, pairs changed {}'.format(cur_iter, i, pair_changed))
                cur_iter += 1
            else:
                non_bound_set = np.nonzero((self._alpha.A > 0) * (self._alpha.A < c))[0]
                for i in non_bound_set:
                    pair_changed += self._innerLoop(i, c, tol)
                    print('non bound, iter {} i: {}, pairs changed {}'.format(cur_iter, i, pair_changed))
                cur_iter += 1

            if entire_set:
                entire_set = False
            elif pair_changed == 0:
                entire_set = True
            print('iteration number {}'.format(cur_iter))

        return self._b, self._alpha

    def _innerLoop(self, i, c, tol):

        ei = self._calEk(i)
        if (self._label[i] * ei < -tol and self._alpha[i] < c) or (self._label[i] * ei > tol and self._label[i] > 0):

            j, ej = self._getJ(i, ei)
            old_alpha_i = self._alpha[i].copy()
            old_alpha_j = self._alpha[j].copy()

            if self._label[i] != self._label[j]:
                l = max(0, self._alpha[j] - self._alpha[i])
                h = min(c, self._alpha[j] - self._alpha[i] + c)
            else:
                l = max(0, self._alpha[j] + self._alpha[i] - c)
                h = min(c, self._alpha[j] + self._alpha[i])

            if l == h:
                print('L == H')
                return 0

            eta = self._input[i, :] * self._input[i, :].T + self._input[j, :] * self._input[j, :].T - 2. * self._input[i, :] * self._input[j, :].T
            if eta <= 0:
                print('eta <= 0')
                return 0
            self._alpha[j] += self._label[j] * (ei - ej) / eta
            self._alpha[j] = self._recAlpha(j, l, h)
            self._updataEk(j)

            if abs(old_alpha_j - self._alpha[j]) < 1e-5:
                print('j does not move enough')
                return 0

            self._alpha[i] += self._label[i] * self._label[j] * (old_alpha_j - self._alpha[j])
            self._updataEk(i)

            b1 = self._b - ei - self._label[i] * (self._alpha[i] - old_alpha_i) * self._input[i, :] * self._input[i, :].T - self._label[j] * (self._alpha[j] - old_alpha_j) * self._input[i, :] * self._input[j, :].T
            b2 = self._b - ei - self._label[i] * (self._alpha[i] - old_alpha_i) * self._input[i, :] * self._input[j, :].T - self._label[j] * (self._alpha[j] - old_alpha_j) * self._input[j, :] * self._input[j, :].T
            if self._alpha[i] > 0 and self._alpha[i] < c:
                self._b = b1
            elif self._alpha[j] > 0 and self._alpha[j] < c:
                self._b = b2
            else:
                self._b = (b1 + b2 ) / 2.
            return 1
        else:
            return 0

    def _updataEk(self, k):
        ek = self._calEk(k)
        self._Ecache[k] = [1, ek]

    def _recAlpha(self, k, l, h):

        if self._alpha[k] < l:
            self._alpha[k] = l
        elif self._alpha[k] > h:
            self._alpha[k] = h
        return self._alpha[k]

    def _getJ(self, i, ei):

        max_k = -1
        max_delta_e = 0
        ej = 0
        self._Ecache[i] = [1, ei]
        valid_j_list = np.nonzero(self._Ecache[:, 0])[0]

        if len(valid_j_list) > 1:
            for k in valid_j_list:
                ek = self._calEk(k)
                if abs(ei - ek) > max_delta_e:
                    max_delta_e = abs(ei - ek)
                    max_k = k
                    ej = ek
            return max_k, ej
        else:
            j = i
            while j == i:
                j = int(np.random.uniform(0, self._input.shape[0]))
            ej = self._calEk(j)
            return j, ej

    def _calEk(self, k):

        pred = float(np.multiply(self._alpha, self._label).T * (self._input * self._input[k, :].T)) + self._b
        return pred - self._label[k]

    def cal_w(self):
        x = self._input
        y = self._label
        alpha = self._alpha
        m, n = x.shape
        w = np.zeros((n, 1))
        for i in range(m):
            w += np.multiply(alpha[i] * y[i], x[i, :].T)
        return w

class svm:

    def __init__(self, input, label, kTup=('lin', 0)):
        self._input = np.mat(input)
        self._label = np.mat(label).reshape((-1, 1))
        self._Ecache = np.mat(np.zeros((self._input.shape[0], 2)))
        self._k = self.kernel_gen(kTup)

        self._alpha = np.mat(np.zeros((self._input.shape[0], 1)))
        self._b = 0
        self._sv = None
        self._sv_label = None


    def run(self, max_iter, c, tol):

        cur_iter = 0
        pair_changed = 0
        entire_set = True

        while (cur_iter < max_iter and pair_changed > 0) or entire_set:
            pair_changed = 0

            if entire_set:
                for i in range(self._input.shape[0]):
                    pair_changed += self._innerLoop(i, c, tol)
                    print('full set, iter {} i: {}, pairs changed {}'.format(cur_iter, i, pair_changed))
                cur_iter += 1
            else:
                non_bound_set = np.nonzero((self._alpha.A > 0) * (self._alpha.A < c))[0]
                for i in non_bound_set:
                    pair_changed += self._innerLoop(i, c, tol)
                    print('non bound, iter {} i: {}, pairs changed {}'.format(cur_iter, i, pair_changed))
                cur_iter += 1

            if entire_set:
                entire_set = False
            elif pair_changed == 0:
                entire_set = True
            print('iteration number {}'.format(cur_iter))

        sv_mask = np.nonzero(self._alpha.A > 0)[0]
        self._sv = self._input[sv_mask]
        self._sv_label = self._label[sv_mask].reshape(-1, 1)
        self._alpha = self._alpha[sv_mask]

        del self._input
        del self._label
        del self._Ecache
        del self._k

        return self._b, self._alpha

    def predict(self, input, kTup):
        ker = np.mat(np.zeros((self._sv.shape[0], 1)))
        if kTup[0] == 'lin':
            ker = self._sv * input.T
        elif kTup[0] == 'rbf':
            for i in range(self._sv.shape[0]):
                delta_row = self._sv[i] - input
                ker[i] = delta_row * delta_row.T
            ker = np.exp(ker / (-1 * kTup[1] ** 2))
        else:
            print('Kernel is not recognized.')
            return -1
        pred = np.multiply(self._alpha, self._sv_label).T * ker + self._b
        return pred


    def kernel_gen(self, kTup):
        input = self._input
        m, n = input.shape
        k = np.mat(np.zeros((m, m)))
        for i in range(m):
            for j in range(m):
                if kTup[0] == 'lin':
                    k[i, j] = input[i, :] * input[j, :].T
                elif kTup[0] == 'rbf':
                    delta_row = input[i, :] - input[j, :]
                    delta = delta_row * delta_row.T
                    k[i, j] = np.exp(delta / (-1 * kTup[1] ** 2))
                else:
                    print('Kernel is not recognized')
        return k

    def _innerLoop(self, i, c, tol):

        ei = self._calEk(i)
        if (self._label[i] * ei < -tol and self._alpha[i] < c) or (self._label[i] * ei > tol and self._label[i] > 0):

            j, ej = self._getJ(i, ei)
            old_alpha_i = self._alpha[i].copy()
            old_alpha_j = self._alpha[j].copy()

            if self._label[i] != self._label[j]:
                l = max(0, self._alpha[j] - self._alpha[i])
                h = min(c, self._alpha[j] - self._alpha[i] + c)
            else:
                l = max(0, self._alpha[j] + self._alpha[i] - c)
                h = min(c, self._alpha[j] + self._alpha[i])

            if l == h:
                print('L == H')
                return 0

            eta = self._k[i, i] + self._k[j, j] - 2. * self._k[i, j]
            if eta <= 0:
                print('eta <= 0')
                return 0
            self._alpha[j] += self._label[j] * (ei - ej) / eta
            self._alpha[j] = self._recAlpha(j, l, h)
            self._updataEk(j)

            if abs(old_alpha_j - self._alpha[j]) < 1e-5:
                print('j does not move enough')
                return 0

            self._alpha[i] += self._label[i] * self._label[j] * (old_alpha_j - self._alpha[j])
            self._updataEk(i)

            b1 = self._b - ei - self._label[i] * (self._alpha[i] - old_alpha_i) * self._k[i, i] - self._label[j] * (self._alpha[j] - old_alpha_j) * self._k[i, j]
            b2 = self._b - ei - self._label[i] * (self._alpha[i] - old_alpha_i) * self._k[i, j] - self._label[j] * (self._alpha[j] - old_alpha_j) * self._k[j, j]
            if self._alpha[i] > 0 and self._alpha[i] < c:
                self._b = b1
            elif self._alpha[j] > 0 and self._alpha[j] < c:
                self._b = b2
            else:
                self._b = (b1 + b2 ) / 2.
            return 1
        else:
            return 0

    def _updataEk(self, k):
        ek = self._calEk(k)
        self._Ecache[k] = [1, ek]

    def _recAlpha(self, k, l, h):

        if self._alpha[k] < l:
            self._alpha[k] = l
        elif self._alpha[k] > h:
            self._alpha[k] = h
        return self._alpha[k]

    def _getJ(self, i, ei):

        max_k = -1
        max_delta_e = 0
        ej = 0
        self._Ecache[i] = [1, ei]
        valid_j_list = np.nonzero(self._Ecache[:, 0])[0]

        if len(valid_j_list) > 1:
            for k in valid_j_list:
                ek = self._calEk(k)
                if abs(ei - ek) > max_delta_e:
                    max_delta_e = abs(ei - ek)
                    max_k = k
                    ej = ek
            return max_k, ej
        else:
            j = i
            while j == i:
                j = int(np.random.uniform(0, self._input.shape[0]))
            ej = self._calEk(j)
            return j, ej

    def _calEk(self, k):

        pred = float(np.multiply(self._alpha, self._label).T * (self._input * self._input[k, :].T)) + self._b
        return pred - self._label[k]

    def cal_w(self):
        x = self._input
        y = self._label
        alpha = self._alpha
        m, n = x.shape
        w = np.zeros((n, 1))
        for i in range(m):
            w += np.multiply(alpha[i] * y[i], x[i, :].T)
        return w