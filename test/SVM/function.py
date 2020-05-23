import numpy as np


def loadData(fileName):
    num_feat = len(open(fileName, 'r').readline().strip().split('\t')) - 1
    x = []
    y = []
    with open(fileName, 'r') as file:
        for line in file.readlines():
            str = line.strip().split('\t')
            cur = []
            for i in range(num_feat):
                cur.append(float(str[i]))
            x.append(cur)
            y.append(float(str[-1]))

    x = np.array(x)
    y = np.array(y)

    x1 = x[:, 0]
    mask = np.argsort(x1)
    x = x[mask]
    y = y[mask]

    return x, y

