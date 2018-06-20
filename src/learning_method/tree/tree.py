# coding=utf-8

import math
from collections import defaultdict
from sortedmap import sortedmap

from src.learning_method.tree.exceptions import DataError
from src.util.define import MIN_FLOAT
import src.util.df as df


def calc_probability(result):
    """calculate probability of result

    :param result: list, for example[yes, no, yes, no]
    :return: dict, for example: {'yes': 0.5, 'no': 0.5}
    """
    res = {}
    total = len(result)
    for item in result:
        if item in res:
            res[item] += 1
        else:
            res[item] = 1
    for key, value in res.items():
        res[key] = value / total
    return res


def calc_entropy(result):
    """Here result is designed as a dict.
    For example, result = {'Yes': 0.6, 'No': 0.4}
    :param result:
    :return: float
    """
    entropy = 0.0
    for value in result.values():
        if math.fabs(value) < 1E-7:
            entropy += 0.0
        else:
            entropy += -value * math.log2(value)
    return entropy


def calc_condition_entropy(features, results):
    """calculate condition entropy of specified feature

    :param features: list
    :param results: list
    :return: float
    """
    total = len(features)
    assert total == len(results)
    condition_map = defaultdict(list)
    for feature, result in zip(features, results):
        condition_map[feature].append(result)
    entropy = 0.0
    len_value = 0
    for value in condition_map.values():
        res = calc_probability(value)
        entropy += len(value) / total * calc_entropy(res)
        len_value += len(value)
    if len_value != total:
        raise DataError(message='data error! sum of length of each feature does not equal total!')
    return entropy


def feature_entropy(features, results):
    """

    :param features: feature list
    :param results: result list
    :return: float
    """
    total = len(features)
    assert total == len(results)
    condition_map = defaultdict(list)
    for feature, result in zip(features, results):
        condition_map[feature].append(result)
    entropy = 0.0
    len_value = 0
    for value in condition_map.values():
        p = len(value) / total
        entropy += -p * math.log2(p)
        len_value += len(value)
    if len_value != total:
        raise DataError(message='data error! sum of length of each feature does not equal total!')
    return entropy


def gain(features, results):
    """
    :param features: list
    :param results: list
    :return: float
    """
    res = calc_probability(results)
    entropy = calc_entropy(res)
    condition_entropy = calc_condition_entropy(features, results)
    return entropy - condition_entropy


def gain_ratio(features, results):
    """
    :param features: list
    :param results: list
    :return: float
    """
    f_entropy = feature_entropy(features, results)
    f_gain = gain(features, results)
    return f_gain - f_entropy


class TreeNode(object):
    """Decision Tree Node
    Each node has two branches, one is yes, the other is no.
    """

    def __init__(self, feature, label=None, root=None, childs=None):
        self.feature = feature
        self.label = label
        self.root = root  # father node
        self.childs = childs  # child TreeNode list


class DecisionTreeMethod(object):
    ID3 = 1
    C45 = 2

    method_map = {
        ID3: gain,
        C45: gain_ratio
    }


class DecisionTree(object):

    def __init__(self, root):
        self.root = root

    @classmethod
    def get_max_key_value(cls, kwargs):
        """

        :param kwargs: dict
        :return:
        """
        tmp_value = MIN_FLOAT
        tmp_key = None
        for key, value in kwargs.items():
            if tmp_value == MIN_FLOAT:
                tmp_value = value
                tmp_key = key
            else:
                if value > tmp_value:
                    tmp_value = value
                    tmp_key = key
        return tmp_key, tmp_value

    @classmethod
    def build(cls, data_train, eps, method=DecisionTreeMethod.ID3):
        """

        :param data_train: pd.DataFrame
        :param eps:
        :param method: ID3, C4.5
        :return: TreeNode
        """
        result = df.get_results(data_train)
        features_list = df.get_features(data_train)
        res = calc_probability(result)

        gain_method = DecisionTreeMethod.method_map.get(method) or gain

        if len(res) == 1:
            labels = list(res.keys())
            node = TreeNode(labels[0])
            return node

        if not features_list:
            key, _ = cls.get_max_key_value(res)
            node = TreeNode(key)
            return node

        gain_map = {}
        for key, value_list in features_list.items():
            gain_map[key] = gain_method(value_list, result)

        max_key, max_gain = cls.get_max_key_value(gain_map)
        if max_gain < eps:
            key, _ = cls.get_max_key_value(res)
            node = TreeNode(key)
            return node
        else:
            root = TreeNode(max_key)
            values = features_list.pop(max_key)
            values = set(values)
            childs = []
            for value in values:
                sub_frame = data_train[data_train[max_key].isin([value])]
                node = cls.build(sub_frame, eps, method)
                childs.append(node)
            root.childs = childs
            return root

    @classmethod
    def in_order(cls, root):
        """Visit a decision tree
        :param root: TreeNode
        :return:
        """
        if root is not None:
            print(root.feature)
            if root.childs:
                for node in root.childs:
                    cls.in_order(node)
