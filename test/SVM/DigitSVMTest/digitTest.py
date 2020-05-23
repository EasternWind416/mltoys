import numpy as np
import os

from ml.SVM import svm


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


def img2vec(filename):
    retVec = np.zeros((1, 1024))
    file = open(filename)
    for i in range(32):
        lineStr = file.readline()
        for j in range(32):
            retVec[0, 32 * i + j] = int(lineStr[j])
    return retVec


def load_img(dir_name):
    labels = []
    print(dir_name)
    train_file_list = os.listdir(dir_name)
    m = len(train_file_list)
    train = np.zeros((m, 1024))
    for i in range(m):
        file_name_str = train_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num = int(file_str.split('_')[0])
        if class_num == 1:
            labels.append(1)
        else:
            labels.append(-1)
        train[i, :] = img2vec(os.path.join(dir_name, file_name_str))
    return np.mat(train), np.mat(labels).reshape(-1, 1)


def testDigits(kTup=('rbf', 10)):
    # 1. 导入训练数据
    input, label = load_img('../data/trainingDigits')

    SVM = svm(input, label, kTup)

    b, alphas = SVM.run(1e4, 200, 1e-4)

    sv_idx = np.nonzero(alphas.A > 0)[0]
    sv_vector = input[sv_idx]
    sv_label = label[sv_idx]
    # print("there are %d Support Vectors" % shape(sv_vector)[0])
    m, n = input.shape
    err_cnt = 0
    for i in range(m):
        predict = SVM.predict(input[i, :], kTup)
        if np.sign(predict) != np.sign(label[i]):
            err_cnt += 1
    print("the training error rate is: %f" % (float(err_cnt) / m))

    # 2. 导入测试数据
    input, label = load_img('../data/trainingDigits')
    err_cnt = 0
    m, n = input.shape
    for i in range(m):
        predict = SVM.predict(input[i, :], kTup)
        if np.sign(predict) != np.sign(label[i]):
            err_cnt += 1
    print("the test error rate is: %f" % (float(err_cnt) / m))


def testRBF(k = 1.3):
    kTup = ('rbf', k)
    input, label = loadData('../data/testSetRBF.txt')
    SVM = svm(input, label, kTup)

    b, alphas = SVM.run(max_iter=1e4, c=200, tol=1e-4)
    m, n = np.shape(input)
    err_cnt = 0
    for i in range(m):
        predict = SVM.predict(input[i, :], kTup)
        if np.sign(predict) != np.sign(label[i]):
            err_cnt += 1
    print("the training error rate is: %f" % (float(err_cnt) / m))

    input, label = loadData('../data/testSetRBF2.txt')
    err_cnt = 0
    m, n = np.shape(input)
    for i in range(m):
        predict = SVM.predict(input[i, :], kTup)
        if np.sign(predict) != np.sign(label[i]):
            err_cnt += 1
    print("the test error rate is: %f" % (float(err_cnt) / m))


if __name__ == '__main__':

    testRBF(0.1)

    # 项目实战
    # 示例: 手写识别问题回顾
    # testDigits(('rbf', 0.1))
    # testDigits(('rbf', 5))
    # testDigits(('rbf', 10))
    # testDigits(('rbf', 50))
    # testDigits(('rbf', 100))
    # testDigits(('lin', 0))
