# coding=utf-8
import numpy as np
from src.util.exceptions import DataError


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


class LogisticModel(object):
    """
    L(w) = \sum_{i=1}^{n} [y_i*(w*x_i) - log(1 + exp(w*x_i))]
    """

    @staticmethod
    def extract(data_train):
        data = data_train.values
        row, column = np.shape(data)
        results = np.reshape(data[:, -1], (row, 1))  # The last column of train data is result.
        features = data[:, 0:column - 1]  # The rest is features.
        return features, results

    def __init__(self, features, results, step, max_iteration):
        """Logistic Model

        :param features: np.array, can be required by `extract` method
        :param results: np.array, can be required by `extract` method
        :param step: factor of each iteration
        :param max_iteration: maximum iterations
        """
        row_f, column_f = np.shape(features)
        row_r = len(results)
        if row_r != row_f:
            raise DataError('Features and results should have the same rows')

        self.weight = np.ones((column_f + 1, 1))
        self.step = step
        self.max_iteration = max_iteration

        self.iterated = 0
        x_addition = np.ones((row_f, 1))
        self.x = np.column_stack((features, x_addition))
        self.features = features
        self.y = results
        self.err = 0

    def _normalize_feature_values(self):
        """If values of features varies several orders of magnitude,
        normalization is necessary for gradient iteration.

        :return:
        """
        self.column_max_values = np.max(self.x, axis=0)
        self.column_min_values = np.min(self.x, axis=0)
        _, column = np.shape(self.features)
        for j in range(column):
            self.x[:, j] = (self.x[:, j] - self.column_min_values[j]) / (
                    self.column_max_values[j] - self.column_min_values[j])

    def _restore_feature_values(self):
        _, column = np.shape(self.features)
        for j in range(column):
            self.x[:, j] = self.x[:, j] * (self.column_max_values[j] - self.column_min_values[j]) + \
                           self.column_min_values[j]

    def _gradient_ascent(self):
        while self.iterated < self.max_iteration:
            res = sigmoid(np.dot(self.x, self.weight))
            self.err = self.y - res
            # change to `self.weight - self.step * np.dot(self.x.transpose(), self.err)` if descent is needed
            self.weight = self.weight + self.step * np.dot(self.x.transpose(), self.err)
            self.iterated += 1

    def build(self):
        return self._gradient_ascent()

    def predict(self, features_predict):
        row_f, column_f = np.shape(features_predict)
        x_addition = np.ones((row_f, 1))
        predict_x = np.column_stack((features_predict, x_addition))
        return sigmoid(np.dot(predict_x, self.weight))
