# coding=utf-8

import src.util.df as df
from src.util.define import MIN_FLOAT
from src.learning_method.tree.util.gain import (
    DecisionTreeMethod, gain
)


class TreeNode(object):
    """Decision Tree Node
    Each node has multi branches.
    """

    def __init__(self, feature=None, label=None, root=None, children=None):
        """
        :param feature: feature name
        :param label: feature label
        :param root: root of current TreeNode
        :param children: list of TreeNode, indicating the children of a TreeNode
        """
        self.feature = feature
        self.label = label
        self.root = root  # father node
        self.children = children  # child TreeNode list


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
            children = []
            for value in values:
                sub_frame = data_train[data_train[max_key].isin([value])]
                node = cls.build(sub_frame, eps, method)
                node.root = root
                children.append(node)
            root.children = children
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
