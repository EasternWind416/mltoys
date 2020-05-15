import numpy as np

from test.LinearRegression.LegoTest.LegoFunction import *
from Main.Regression.RidgeRegress import RR
from Main.Regression.LinearRegress import LR


X = []
Y = []

setDataCollect(X, Y)
crossValid(X, Y, 10)
#
# X = np.array(X)
# Y = np.array(Y)
#
# Xmean = X.mean(axis=0)
# Ymean = Y.mean()
# Xvar = X.var(axis=0)
# X = (X - Xmean) / Xvar
# Y = Y - Ymean
#
# theta, pred = RR().regress(X, Y, np.exp(0))
# # pred = LR(X, Y).LWR()
# print('err is: ', ((pred - Y) ** 2).sum())
