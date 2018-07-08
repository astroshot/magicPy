# coding=utf-8
import pandas as pd
from unittest import TestCase, main
# import test_init
from src.learning_method.bayes import bayes


class TestBayes(TestCase):
    data_train = pd.read_csv('tests/learning_method/tree/test_credit.csv')

    def test_build(self):
        bmodel = bayes.NaiveBayes(self.data_train)
        bmodel.build(method=2)
        


if __name__ == '__main__':
    main()
