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


class DecisionTreeMethod(object):
    ID3 = 1
    C45 = 2

    method_map = {
        ID3: gain,
        C45: gain_ratio
    }
