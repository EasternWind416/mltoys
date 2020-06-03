import numpy as np
import operator


def classify(input_x, input, label, k):
    diff = input - input_x
    diff = diff ** 2
    diff = np.sum(diff, axis=1)
    diff = diff ** .5

    sorted_diff_indx = np.argsort(diff)
    class_cnt = {}
    for i in range(k):
        vote_i_label = int(label[sorted_diff_indx[i]])
        if class_cnt.get(vote_i_label) is None:
            class_cnt[vote_i_label] = 1
        else:
            class_cnt[vote_i_label] += 1
    sorted_class_cnt = sorted(class_cnt.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_cnt[0][0]