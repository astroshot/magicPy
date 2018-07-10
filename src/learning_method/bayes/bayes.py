# coding=utf-8
import numpy as np
import src.util.df as df
from collections import Counter


class Method(object):
    MAXIMUM_LIKELIHOOD = 1
    BAYES_METHOD = 2


class NaiveBayes(object):
    """
    model
    {
        "No": {
            "Age": {
                "Young": 0.5,
                "Medium": 0.3333333333333333,
                "Old": 0.16666666666666666
            },
            "Job": {
                "No": 1.0
            },
            "House": {
                "No": 1.0
            },
            "Credit": {
                "Normal": 0.6666666666666666,
                "Good": 0.3333333333333333
            }
        },
        "Yes": {
            "Age": {
                "Young": 0.2222222222222222,
                "Medium": 0.3333333333333333,
                "Old": 0.4444444444444444
            },
            "Job": {
                "Yes": 0.5555555555555556,
                "No": 0.4444444444444444
            },
            "House": {
                "No": 0.3333333333333333,
                "Yes": 0.6666666666666666
            },
            "Credit": {
                "Good": 0.4444444444444444,
                "Normal": 0.1111111111111111,
                "Very Good": 0.4444444444444444
            }
        }
    }
    """

    def __init__(self, data_train, result=None, model=None):
        """Naive Bayes model
        :param data_train: pandas.DataFrame
        :param result: result of data_train
        :param model: dict of propabilities
        """
        self.data_train = data_train
        self.result = result
        self.model = model
        self.features_list = df.get_features(self.data_train)

    def calc_feature_items(self):
        features_items = {}
        for key, value_list in self.features_list.items():
            counter = Counter(value_list)
            features_items[key] = counter
        return features_items

    def calc_feature_propability(self, feature, values, smooth=False, factor=1):
        if not smooth:
            return df.calc_probability(values)
        feature_values_set = set(values)
        full_values_set = set(self.features_list[feature])
        null_list = full_values_set - feature_values_set
        if not null_list:
            return df.calc_probability(values, smooth, factor)

        total = len(values)
        counter = Counter(values)
        n = len(full_values_set)
        for key, value in counter.items():
            counter[key] = (value + factor) / (total + factor * n)
        for key in null_list:
            counter[key] = factor / (total + factor * n)
        return counter

    def build(self, method=Method.MAXIMUM_LIKELIHOOD):
        """Stores bayes propability in dict. for example:

        """
        data_train = self.data_train
        result = df.get_results(data_train)
        smooth = method == Method.BAYES_METHOD
        res = df.calc_probability(result, smooth)
        column_names = df.get_column_names(data_train)
        assert len(column_names) > 0
        result_row_name = column_names[-1]
        bayes_model = {}
        for key in res:
            bayes_model[key] = {}
            sub_frame = data_train[data_train[result_row_name].isin([key])]
            features_map = df.get_features(sub_frame)
            for feature, values in features_map.items():
                assert type(values) == list
                feature_res = self.calc_feature_propability(feature, values, smooth)
                bayes_model[key][feature] = feature_res
        self.result = res
        self.model = bayes_model

    def predict(self, data_frame):
        features = df.get_features(data_frame)
        columns = df.get_column_names(data_frame)
        predict_values = {}
        append_result = []

        for _, row in data_frame.iterrows():
            for y_value in self.result:
                p = self.result[y_value]
                p_features = 1
                for column in columns:
                    p_features *= self.model[y_value][column].get(row[column], 0)
                predict_values[y_value] = p * p_features
            max_value, max_key = max(zip(predict_values.values(), predict_values.keys()))
            append_result.append(max_key)
        data_frame['Result'] = append_result
        return data_frame