# coding=utf-8
import matplotlib as plt
import numpy as np
import pandas as pd
import seaborn as sns

data_train = pd.read_csv('./data/train.csv')
data_test = pd.read_csv('./data/test.csv')


def list_data():
    # 数据整体情况
    data_train.info()
    # 数值类型数据分布
    data_train.describe()
    # 离散类型的数据分布
    data_train.describe(include=[np.object])

    # 离散型特征与Survived之间的关联性
    # Pclass Survived
    data_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

    data_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    data_train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    data_train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    data_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
