from test.LinearRegression.LegoTest.LegoFunction import *
from ml.Regress import RidgeRegress

X = []
Y = []

setDataCollect(X, Y)
# crossValid(X, Y, 10)
#
X = np.array(X)
Y = np.array(Y)

Xmean = X.mean(axis=0)
Ymean = Y.mean()
Xvar = X.var(axis=0)
X = (X - Xmean) / Xvar
# Y = Y - Ymean

theta, pred = RidgeRegress().regress(X, Y, np.exp(0))
# pred = LR(X, Y).LWR()
print('err is: ', (abs(pred - Y)).sum())
