import numpy as np

from ml.DecisionTree import *


def create_data():
    data = np.array([[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']])
    x = data[:, :2]
    y = data[:, -1]
    label = np.array(['no surfacing', 'flippers'])
    return x, y, label


if __name__ == '__main__':
    x, y, label = create_data()
    dt = DecisionTree(x, y, label)
    pred = dt.run(('0', '1'))
    if pred == 'yes':
        print('yes')
    elif pred == 'no':
        print('no')