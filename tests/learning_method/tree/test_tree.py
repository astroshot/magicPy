# coding=utf-8

import pandas as pd
from unittest import TestCase, main
import test_init
from src.learning_method.tree import tree


class TestTree(TestCase):
    data_train = pd.read_csv('tests/learning_method/tree/test_credit.csv')

    def test_calc_probability(self):
        result = self.data_train['Classification'].values.tolist()
        res = tree.calc_probability(result)
        self.assertEqual(type(res), dict)
        entropy = tree.calc_entropy(res)
        A1 = self.data_train['Age'].values.tolist()
        condition_entropy = tree.calc_condition_entropy(A1, result)
        gain1 = tree.gain(A1, result)
        self.assertEqual(gain1, entropy - condition_entropy)


if __name__ == '__main__':
    main()
