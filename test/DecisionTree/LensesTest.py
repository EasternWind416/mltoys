import numpy as np

from ml.DecisionTree import DecisionTree


def load_data(fine_name):
    with open(fine_name, 'r') as file:
        lenses = np.array([inst.strip().split('\t') for inst in file.readlines()])
        x = lenses[:, : 4]
        y = lenses[:, -1]
        label = np.array(['age', 'prescript', 'astigmatic', 'tearRate'])
    return x, y, label


if __name__ == '__main__':
    x, y, label = load_data('./data/lenses.txt')
    dt = DecisionTree(x, y, label)
    pred = dt.run(('young', 'myope', 'no', 'normal'))
    print('---', pred)