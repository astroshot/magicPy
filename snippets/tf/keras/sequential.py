# coding: utf-8

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical


def sequential():
    model = Sequential()
    # Dense 全连接层
    model.add(Dense(32, activation='relu', input_dim=100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    data = np.random.random((1000, 100))
    labels = np.random.randint(2, size=(1000, 1))

    model.fit(data, labels, epochs=10, batch_size=32)
    # 获取 model 每一层 layer 的权重
    model.get_weights()

    return model


def build_model():
    model = Sequential()

    # stacking layers
    model.add(Dense(units=32, activation='relu', input_dim=100))
    model.add(Dense(units=10, activation='softmax'))

    # config its learning process
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # For a multi-class classification problem
    # model.compile(optimizer='rmsprop',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    #
    # # For a binary classification problem
    # model.compile(optimizer='rmsprop',
    #               loss='binary_crossentropy',
    #               metrics=['accuracy'])
    #
    # # For a mean squared error regression problem
    # model.compile(optimizer='rmsprop',
    #               loss='mse')
    return model


def training(model):
    data = np.random.random((1000, 100))
    labels = np.random.randint(10, size=(1000, 1))

    one_hot_labels = to_categorical(labels, num_classes=10)
    model.fit(data, one_hot_labels, epochs=10, batch_size=32)
    return model


if __name__ == '__main__':
    model = sequential()
    print(model.summary())
    model = build_model()
    model = training(model)
