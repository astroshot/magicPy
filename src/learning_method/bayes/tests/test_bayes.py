# coding=utf-8
from unittest import TestCase, main

import pandas as pd

import os
import sys

join = os.path.join
base = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
base = os.path.dirname(base)
sys.path.append(os.getcwd())
sys.path.append('src/learning_method/bayes')


class TestBayes(TestCase):
    data_train = pd.read_csv('src/learning_method/tree/tests/test_credit.csv')

    def test_build(self):
        from src.learning_method.bayes import bayes

        bmodel = bayes.NaiveBayes(self.data_train)
        bmodel.build(method=2)
        self.assertEqual(len(bmodel.model), 2)

    def test_predict(self):
        from src.learning_method.bayes import bayes

        predict_data = pd.read_csv('src/learning_method/bayes/tests/credit_predict.csv')
        bmodel = bayes.NaiveBayes(self.data_train)
        bmodel.build(method=2)
        result_data = bmodel.predict(predict_data)
        predict_list = result_data['Result'].values.tolist()
        self.assertEqual(predict_list, ['No', 'No', 'Yes'])


if __name__ == '__main__':
    main()
