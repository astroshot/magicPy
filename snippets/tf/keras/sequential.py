# coding: utf-8

from keras.models import Sequential
from keras.layers import Dense
import numpy as np


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


if __name__ == '__main__':
    model = sequential()
    print(model.summary())
