# coding=utf-8


from collections import defaultdict, Counter
import pandas as pd


def get_column_names(data_frame):
    """get column names of data_frame
    data_frame is what pd.read_csv() returns, which has rows and columns,
    and the last column is result.
    :param data_frame: pd.DataFrame
    :return: list, name of each column name as feature name
    """
    return [col for col in data_frame]


def get_results(data_frame):
    """Using the last column as result
    :param data_frame: pandas.DataFrame
    :return: list of values
    """
    columns = get_column_names(data_frame)
    return data_frame[data_frame.columns[len(columns)-1]].values.tolist() if columns else []


def get_features(data_frame):
    """
    :param data_frame: pandas.DataFrame
    :return: map, using name of columns as key, list of feature values as value
    """
    columns = get_column_names(data_frame)
    features_list = {}
    for i, name in enumerate(columns):
        if i < len(columns) - 1:
            features_list[name] = data_frame[data_frame.columns[i]].values.tolist()
    return features_list


def calc_probability(result, smooth=False, factor=1):
    """calculate probability of result

    :param result: list, for example[yes, no, yes, no]
    :param smooth: boolean, if Laplace smoothing is needed
    :param factor: smoothing factor, default value is 1
    :return: dict, for example: {'yes': 0.5, 'no': 0.5}
    """
    if smooth:  # Laplace smoothing
        if not factor:
            factor = 1

    total = len(result)
    counter = Counter(result)
    n = len(counter)

    if smooth:
        for key, value in counter.items():
            counter[key] = (value + factor) / (total + factor * n)
    else:
        for key, value in counter.items():
            counter[key] = value / total

    return counter
