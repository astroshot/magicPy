# coding=utf-8
import numpy as np
import pandas as pd

from unittest import TestCase, main

import os
import sys

join = os.path.join
base = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
base = os.path.dirname(base)
sys.path.append(os.getcwd())
sys.path.append('src/learning_method/logistic')


class TestLogistic(TestCase):
    data_train = pd.read_csv('src/learning_method/logistic/tests/logistic_test.csv', sep='\t')

    def test_build(self):
        from src.learning_method.logistic.logistic import LogisticModel
        features, results = LogisticModel.extract(self.data_train)
        model = LogisticModel(features, results, 1e-3, 500)
        model.build()
        weight = np.array([[0.47911342], [-0.61587567], [4.11858007]])
        self.assertEqual(weight, model.weight)


if __name__ == '__main__':
    main()
