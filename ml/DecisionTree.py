import numpy as np


def cal_h(y):
    y_count = {}
    for y_eval in y:
        if y_eval not in y_count.keys():
            y_count[y_eval] = 1
        else:
            y_count[y_eval] += 1
    h = 0.
    for key in y_count:
        prob = y_count[key] / float(len(y))
        h -= prob * np.log2(prob)
    return h


def majority(y):
    major_num = 0
    major = None
    c = set(y)
    for example in c:
        if y.count(example) > major_num:
            major_num = y.count(example)
            major = example
    return major


def split(x, y, col, val):
    sub_x = []
    sub_y = []
    for k, x_eval in enumerate(x):
        x_eval = list(x_eval)
        if x_eval[col] == val:
            reduced_x = x_eval[: col]
            reduced_x.extend(x_eval[col + 1:])
            sub_x.append(reduced_x)
            sub_y.append(y[k])
    return sub_x, sub_y


def find_best_feature(x, y):
    num_feature = len(x[0])
    base_h = cal_h(y)
    max_gain_h = 0.
    max_gain_h_feature = -1

    for i in range(num_feature):
        h = 0.
        value = set(x[i])
        for val in value:
            sub_x, sub_y = split(x, y, i, val)
            h += cal_h(sub_y)
        gain_h = base_h - h
        print('now gain entropy = {}, the feature num = {}'.format(gain_h, i))
        if gain_h > max_gain_h:
            max_gain_h = gain_h
            max_gain_h_feature = i
    print('\n---\nthis round best feature is ', max_gain_h_feature, '\n---\n')
    return max_gain_h_feature


class DecisionTree:
    def __init__(self, train_x, train_y, train_label):
        self._x = list(train_x)
        self._y = list(train_y)
        self._label = list(train_label)

        self._tree = self.create_tree(self._x, self._y, self._label.copy())

    def create_tree(self, x, y, label):
        class_list = [y_eval for y_eval in y]
        '''如果只有一个类别，那么直接返回'''
        if class_list.count(class_list[0]) == len(class_list):
            return class_list[0]

        '''如果特征不够了，也需要返回'''
        if len(x) == 0:
            return majority(class_list)

        best_feature = find_best_feature(x, y)
        best_feature_str = label[best_feature]
        feature_val = set([feature[best_feature] for feature in x])

        tree = {best_feature_str: {}}
        del label[best_feature]

        for val in feature_val:
            rest_label = label[:]
            sub_x, sub_y = split(x, y, best_feature, val)
            tree[best_feature_str][val] = self.create_tree(sub_x, sub_y, rest_label)
        return tree

    def classify(self, tree, test_val):
        root = list(tree.keys())[0]
        root_dict = tree[root]

        feature_indx = self._label.index(root)
        key = test_val[feature_indx]
        value_of_feature = root_dict[key]

        print('\n===\nroot: ', root, ', dict: ', root_dict, ', key = ', key, ' >>> ', value_of_feature, '\n===\n')
        if isinstance(value_of_feature, dict):
            class_label = self.classify(value_of_feature, test_val)
        else:
            class_label = value_of_feature
        return class_label

    def run(self, test_val):
        return self.classify(self._tree, test_val)

    def get_tree(self):
        return self._tree
