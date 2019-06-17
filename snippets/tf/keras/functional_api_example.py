# coding=utf-8
"""
"""

from keras.models import Model
from keras.layers import Input, Dense


def example():
    inputs = Input((784,))
    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # model.fit(data, labels)  # starts training


if __name__ == '__main__':
    example()
