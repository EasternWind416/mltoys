import numpy as np
from random import seed, randrange, random

from ml.Ensemble import RandomForest


def is_digit(str):
    try:
        float(str)
    except:
        return False
    else:
        return True


def load_data(file_name):
    x = []
    y = []
    with open(file_name, 'r') as file:
        for line in file.readlines():
            if not line:
                continue
            line_arr = []
            cur_x = []
            for feature in line.split(','):
                str = feature.strip()
                if is_digit(str):
                    cur_x.append(float(str))
                else:
                    y.append(str)
            x.append(cur_x)
    return x, y


def cross_valid_split(x, y, n_folds):
    sub_x = []
    sub_y = []
    x_copy = x.copy()
    y_copy = y.copy()
    fold_size = len(x) / n_folds
    for i in range(n_folds):
        fold_x = []
        fold_y = []
        while len(fold_x) < fold_size:
            index = randrange(len(x_copy))
            fold_x.append(x_copy[index])
            fold_y.append(y_copy[index])
        sub_x.append(fold_x)
        sub_y.append(fold_y)
    return sub_x, sub_y


def evaluate(x, y, n_folds, *args):
    fold_x, fold_y = cross_valid_split(x, y, n_folds)
    scores = []
    for i in range(n_folds):
        train_x, train_y = list(fold_x), list(fold_y)
        train_x.remove(fold_x[i])
        train_y.remove(fold_y[i])
        train_x, train_y = sum(train_x, []), sum(train_y, [])

        valid_x, valid_y = fold_x[i], fold_y[i]

        rf = RandomForest(train_x, train_y, *args)
        preds = rf.predict(valid_x)
        acc = acc_metric(preds, valid_y)
        scores.append(acc)
    return scores


def acc_metric(preds, y):
    correct = 0
    for i in range(len(y)):
        if preds[i] == y[i]:
            correct += 1
    return correct / float(len(y))


if __name__ == '__main__':
    x, y = load_data('./data/sonar-all-data.txt')
    print('x len is: {}, y len is: {}'.format(len(x), len(y)))
    n_folds = 5
    max_depth = 20
    min_size = 1
    sample_size = 1.
    n_features = 15
    for n_trees in [1, 10, 20]:
        scores = evaluate(x, y, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
        seed(1)
        print('random=', random())
        print('\nTrees: %d' % n_trees)
        print('Scores: %s' % scores)
        print('Mean Accuracy: {}%'.format(sum(scores) / float(len(scores))))