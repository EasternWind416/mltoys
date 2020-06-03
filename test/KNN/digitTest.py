import numpy as np
import os

from ml.KNN import *


def img2vec(file_name):
    ret_vec = np.zeros((1, 1024))
    with open(file_name, 'r') as file:
        for i in range(32):
            line = file.readline()
            for j in range(32):
                ret_vec[0, 32 * i + j] = int(line[j])
    return ret_vec


if __name__ == '__main__':
    label = []
    train_file_list = os.listdir('./data/trainingDigits')
    m = len(train_file_list)
    train = np.zeros((m, 1024))
    for i in range(m):
        file_name_str = train_file_list[i]
        file_name = file_name_str.split('.')[0]
        class_num_str = int(file_name.split('_')[0])
        label.append(class_num_str)
        train[i, :] = img2vec(os.path.join('./data/trainingDigits', file_name_str))

    test_file_list = os.listdir('./data/testDigits')
    m = len(test_file_list)
    err_cnt = 0
    for i in range(m):
        file_name_str = test_file_list[i]
        file_name = file_name_str.split('.')[0]
        class_num_str = int(file_name.split('_')[0])
        test = img2vec(os.path.join('./data/testDigits', file_name_str))
        pred = classify(test, train, label, 3)
        print('the pred = {}, the answer = {}'.format(pred, class_num_str))
        if pred != class_num_str:
            err_cnt += 1
    print('\nthe total number of errors is: ', err_cnt)
    print('\nthe total error rate is: ', err_cnt / m)
