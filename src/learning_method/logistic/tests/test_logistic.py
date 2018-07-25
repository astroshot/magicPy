# coding=utf-8
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from src.learning_method.logistic.logistic import LogisticModel

data_train = pd.read_csv('src/learning_method/logistic/tests/logistic_test.csv', sep='\t')


def main():
    features, results = LogisticModel.extract(data_train)
    model = LogisticModel(features, results, 1e-3, 500)
    model.build()
    # weight = np.array([[0.48007329], [-0.6168482], [4.12414349]])
    print(model.weight)
    # plot train data
    data_array = data_train.values
    plt.figure()
    plt.scatter(data_array[:, 0], data_array[:, 1], c=data_array[:, 2], alpha=0.4)
    x_line = np.linspace(-3, 3, 51)
    y_line = (-model.weight[2] - model.weight[0] * x_line) / model.weight[1]
    plt.plot(x_line, y_line)
    plt.show()


if __name__ == '__main__':
    main()
