# coding=utf-8


from collections import defaultdict
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
    columns = get_column_names(data_frame)
    return data_frame[data_frame.columns[len(columns)-1]].values.tolist() if columns else []


def get_features(data_frame):
    columns = get_column_names(data_frame)
    features_list = {}
    for i, name in enumerate(columns):
        if i < len(columns) - 1:
            features_list[name] = data_frame[data_frame.columns[i]].values.tolist()
    return features_list
