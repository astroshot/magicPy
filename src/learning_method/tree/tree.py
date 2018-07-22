# coding=utf-8

import math
from collections import defaultdict

import src.util.df as df
from src.util.exceptions import DataError
from src.util.define import MIN_FLOAT, EPS


def calc_entropy(result):
    """Calculate entropy of result. Here result is returned as a dict.
    For example, result = {'Yes': 0.6, 'No': 0.4}
    :param result:
    :return: float
    """
    entropy = 0.0
    for value in result.values():
        if math.fabs(value) < EPS:
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
        res = df.calc_probability(value)
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
    """gain used for ID3
    :param features: list
    :param results: list
    :return: float
    """
    res = df.calc_probability(results)
    entropy = calc_entropy(res)
    condition_entropy = calc_condition_entropy(features, results)
    return entropy - condition_entropy


def gain_ratio(features, results):
    """gain ratio used for C4.5
    :param features: list
    :param results: list
    :return: float
    """
    f_entropy = feature_entropy(features, results)
    f_gain = gain(features, results)
    return f_gain - f_entropy


class TreeNode(object):
    """Decision Tree Node
    Each node has multi branches.
    """

    def __init__(self, feature=None, label=None, root=None, childs=None):
        """
        :param feature: feature name
        :param label: feature label
        :param root: root of current TreeNode
        :param childs: list of TreeNode, indicating the childs of a TreeNode
        """
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
        """Training data is based on DataFrame in pandas.
        :param root: TreeNode
        """
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
        res = df.calc_probability(result)

        gain_method = DecisionTreeMethod.method_map.get(method) or gain

        if len(res) == 1:
            labels = list(res.keys())
            node = TreeNode(label=labels[0])
            return node

        if not features_list:
            key, _ = cls.get_max_key_value(res)
            node = TreeNode(label=key)
            return node

        gain_map = {}
        for key, value_list in features_list.items():
            gain_map[key] = gain_method(value_list, result)

        max_key, max_gain = cls.get_max_key_value(gain_map)
        if max_gain < eps:
            key, _ = cls.get_max_key_value(res)
            node = TreeNode(label=key)
            return node
        else:
            root = TreeNode(feature=max_key)
            values = features_list.pop(max_key)
            values = set(values)
            childs = []
            for value in values:
                sub_frame = data_train[data_train[max_key].isin([value])]
                node = cls.build(sub_frame, eps, method)
                node.root = root
                childs.append(node)
            root.childs = childs
            return root

    @classmethod
    def in_order(cls, root):
        """Visit a decision tree by in_order
        :param root: TreeNode
        :return:
        """
        if root is not None:
            print(root.feature)
            if root.childs:
                for node in root.childs:
                    cls.in_order(node)
