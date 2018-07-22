# coding=utf-8
import numpy as np
import pandas as pd

from unittest import TestCase, main
from src.learning_method.logistic.logistic import LogisticModel


class TestLogistic(TestCase):
    data_train = pd.read_csv('src/learning_method/logistic/test/logistic_test.csv', sep='\t')

    def test_build(self):
        features, results = LogisticModel.extract(self.data_train)
        model = LogisticModel(features, results, 1e-3, 500)
        model.build()
        weight = np.array([[0.47911342], [-0.61587567], [4.11858007]])
        self.assertEqual(weight, model.weight)


if __name__ == '__main__':
    main()
