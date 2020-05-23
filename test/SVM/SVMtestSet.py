import numpy as np
import matplotlib.pyplot as plt

from test.SVM.function import *
from ml.SVM import svm_smo_nk_simple
from ml.SVM import svm_smo_nonkernel

fileName = './data/testSet.txt'
x, y = loadData(fileName)

# svm = svm_smo_nk_simple()
# b, alphas = svm.run(x, y, .6, 1e-3, 40)
# w = svm.cal_w()

svm_sys = svm_smo_nonkernel(x, y)
b, alphas = svm_sys.run(40, .6, 1e-3)
w = svm_sys.cal_w()
x1 = [max(x[:, 0]), min(x[:, 0])]
x2 = [- (b[0, 0] + w[0, 0] * i) / w[1, 0] for i in x1]

plt.plot(x[y == 1, 0], x[y == 1, 1], '.', label='class 1')
plt.plot(x[y == -1, 0], x[y == -1, 1], '.', label='class -1')
plt.plot(x1, x2)
for i in range(len(alphas)):
    if alphas[i] > 0.:
        plt.plot(x[i, 0], x[i, 1], 'ro')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()
