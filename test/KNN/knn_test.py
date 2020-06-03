import numpy as np
import matplotlib.pylab as plt

from ml.KNN import *


def load_data(file_name):
    num_feature = len(open(file_name).readline().strip().split('\t')) - 1
    input = []
    label = []
    with open(file_name, 'r') as file:
        for line in file.readlines():
            line_str = line.strip().split('\t')
            cur_x = []
            for i in range(num_feature):
                cur_x.append(float(line_str[i]))
            input.append(cur_x)
            label.append(float(line_str[-1]))
    return np.array(input), np.array(label)


def norm(input):
    min_input = np.min(input, axis=0)
    max_input = np.max(input, axis=0)
    ranges = max_input - min_input
    input_norm = input - min_input
    input_norm = input_norm / ranges
    return input_norm


def classifyPerson():
    res_list = ['not at all', 'a little dosed', 'great dosed']
    f1 = float(input('frequent filer miles earned per year: '))
    f2 = float(input('percentage of time spent playing video games: '))
    f3 = float(input('liters of ice cream consumed per year: '))
    x, y = load_data('./data/datingTestSet2.txt')
    x = norm(x)
    in_x = np.array([f1, f2, f3])
    res = classify(in_x, x, y, 3)
    print('you will probably like this person: ', res_list[res - 1])



if __name__ == '__main__':
    # classifyPerson()

    file_name = './data/datingTestSet2.txt'
    input, label = load_data(file_name)
    input = norm(input)

    train_valid_ratio = .1
    num_valid = int(train_valid_ratio * input.shape[0])
    print('# valid = ', num_valid)

    err_cnt = 0.
    for i in range(num_valid):
        indx = int(np.random.uniform(0, input.shape[0]))
        classifier_res = classify(input[indx, :], input[num_valid:, :], label[num_valid:], 3)
        print('the classifier res is: ', classifier_res, '- the real answer is: ', label[indx])
        if classifier_res != label[indx]:
            err_cnt += 1.
    print('total error rate is: ', err_cnt / input.shape[0])



    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(input[:, 0], input[:, 1], 3 * label,  label)
    # plt.show()