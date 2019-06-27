# coding=utf-8
"""
"""

from keras.models import Model
from keras.layers import Input, Dense, Embedding, LSTM, concatenate


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


def complex_model():
    """
                        main_input(InputLayer)
                            |
                        embedding(Embedding)
                            |
    aux_input(InputLayer)  LSTM
            \              / \
              Merge1(Merge)      aux_output(Dense)
                    |
              dense1 (Dense)
                    |
              dense2 (Dense)
                    |
              dense3 (Dense)
                    |
             main_output(Dense)
    :return:
    """
    main_input = Input(shape=(100,), dtype='int32', name='main_input')
    x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)
    lstm_output = LSTM(32)(x)
    aux_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_output)

    aux_input = Input(shape=(5,), name='aux_input')
    x = concatenate([lstm_output, aux_input])
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    main_output = Dense(1, activation='sigmoid', name='main_output')(x)
    model = Model(inputs=[main_input, aux_input], outputs=[main_output, aux_output])
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', loss_weights=[1.0, 0.2])
    print(model.summary())


if __name__ == '__main__':
    example()
    complex_model()
