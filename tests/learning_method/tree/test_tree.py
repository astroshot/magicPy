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

    def test_build_ID3(self):
        root = tree.DecisionTree.build(self.data_train, eps=0.1)
        self.assertEqual(root.feature, 'House')
        self.assertEqual(len(root.childs), 2)
        self.assertEqual(root.childs[0].childs, None)
        self.assertEqual(len(root.childs[1].childs), 2)


if __name__ == '__main__':
    main()
