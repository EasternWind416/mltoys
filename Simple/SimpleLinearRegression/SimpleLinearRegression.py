import numpy as np


def get_theta(x, y, W=None):
    '''

    :param x: is a tensor with shape (N, M), which N is the # of data, M is the degrees of polynomial;
    :param y: is the ground truth with shape (N,);
    :param W: is the weighted matrix;
    :return: parameter theta.
    '''
    if W is None:
        return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    else:
        return np.linalg.inv(x.T.dot(W).dot(x)).dot(x.T).dot(W).dot(y)


def LWR(x, y, W=None):
    '''

    :param x: is a tensor with shape (N, M-1), which is lack of interception;
    :param y: is the ground truth with shape (N,);
    :param W: is a function to construct weighted matrix, which should have format like W(x, x_eval);
    :return: prediction.
    '''
    X0 = np.ones(x.shape[0])
    X = np.stack([X0, x], axis=1)
    preds = []

    if W is None:
        '''Then LWR is the normal LR'''
        theta = get_theta(x, y, W)
        preds = X.dot(theta)
    else:
        '''Then LWR'''
        for idx, X_eval in enumerate(X):
            w = W(x, X_eval)
            theta = get_theta(x, y, W)
            pred = X_eval.dot(X)
            preds.append(pred)
        preds = np.array(preds)

    return preds