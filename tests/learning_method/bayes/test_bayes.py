# coding=utf-8
from unittest import TestCase, main

import pandas as pd
import test_init

from src.learning_method.bayes import bayes


class TestBayes(TestCase):
    data_train = pd.read_csv('tests/learning_method/tree/test_credit.csv')

    def test_build(self):
        bmodel = bayes.NaiveBayes(self.data_train)
        bmodel.build(method=2)
        self.assertEqual(len(bmodel.model), 2)

    def test_predict(self):
        predict_data = pd.read_csv('tests/learning_method/bayes/credit_predict.csv')
        bmodel = bayes.NaiveBayes(self.data_train)
        bmodel.build(method=2)
        result_data = bmodel.predict(predict_data)
        predict_list = result_data['Result'].values.tolist()
        self.assertEqual(predict_list, ['No', 'No', 'Yes'])


if __name__ == '__main__':
    main()
