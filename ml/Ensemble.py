import numpy as np
from random import randrange


class RandomForest:
    def __init__(self, train_x, train_y, max_depth, min_size, sample_size, n_trees, n_features):
        self._dataset = []
        for i in range(len(train_x)):
            cur_data = list(train_x[i])
            cur_data.append(train_y[i])
            self._dataset.append(cur_data)

        self._arg = {'depth': max_depth,
                     'size': min_size,
                     'sample_size': sample_size,
                     'trees': n_trees,
                     'features': n_features}

        self._trees = []
        for i in range(self._arg['trees']):
            dataset = self.subsample(self._dataset, self._arg['sample_size'])
            tree = self.build_tree(dataset, self._arg['depth'], self._arg['size'], self._arg['features'])
            self._trees.append(tree)

    def predict(self, test):
        trees = self._trees
        preds = [self.bagging(row, trees) for row in test]
        return preds

    def bagging(self, row, trees):
        preds = [self.single_predict(row, tree) for tree in trees]
        return max(set(preds), key=preds.count)

    def single_predict(self, row, node):
        if row[node['feature']] < node['value']:
            if isinstance(node['left'], dict):
                return self.single_predict(row, node['left'])
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.single_predict(row, node['right'])
            else:
                return node['right']

    def build_tree(self, dataset, max_depth, min_size, n_features):
        root = self.get_split(dataset, n_features)
        self.split(root, max_depth, min_size, n_features, 1)
        return root

    def split(self, node, max_depth, min_size, n_features, depth):
        left, right = node['groups']
        del node['groups']
        if not left or not right:
            node['left'] = node['right'] = self.terminal(left + right)
            return
        if depth >= max_depth:
            node['left'], node['right'] = self.terminal(left), self.terminal(right)
            return
        if len(left) <= min_size:
            node['left'] = self.terminal(left)
        else:
            node['left'] = self.get_split(left, n_features)
            self.split(node['left'], max_depth, min_size, n_features, depth + 1)
        if len(right) <= min_size:
            node['right'] = self.terminal(right)
        else:
            node['right'] = self.get_split(right, n_features)
            self.split(node['right'], max_depth, min_size, n_features, depth + 1)

    def terminal(self, group):
        outcomes = [row[-1] for row in group]
        return max(outcomes, key=outcomes.count)

    def get_split(self, dataset, n_features):
        class_val = set([data[-1] for data in dataset])
        features = []
        best_feature, best_val, best_g, best_groups = 999, None, float('inf'), None
        while len(features) < n_features:
            index = randrange(len(dataset[0]) - 1)
            features.append(index)

        for feature in features:
            for cur_data in dataset:
                groups = self.test_split(dataset, feature, cur_data[feature])  # groups = (left, right)
                g = self.cal_g(groups, class_val)
                if g < best_g:
                    best_feature, best_val, best_g, best_groups = feature, cur_data[feature], g, groups
        return {'feature': best_feature, 'value': best_val, 'gini': best_g, 'groups': best_groups}

    def cal_g(self, groups, class_val):
        g = 0.
        size = 0
        for group in groups:
            size += len(group)
        for group in groups:
            sub_g = 0.
            sub_size = len(group)
            if sub_size == 0:
                continue
            for val in class_val:
                p = [row[-1] for row in group].count(val) / float(sub_size)
                sub_g += p * (1 - p)
            g += sub_size / float(size) * sub_g
        return g

    def test_split(self, dataset, feature, val):
        left, right = [], []
        for data in dataset:
            if data[feature] < val:
                left.append(data)
            else:
                right.append(data)
        return left, right

    def subsample(self, dataset, sample_size):
        sub_data = []
        size = round(len(dataset) * sample_size)
        while len(sub_data) < size:
            index = randrange(size)
            sub_data.append(dataset[index])
        return sub_data
