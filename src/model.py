import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.python.keras.layers import Conv2D, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM


def get_cnn_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                     activation='relu', input_shape=(28, 28, 1)))

    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                     activation='relu'))

    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation="softmax"))

    # Compile model
    optimizer = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'],
    )

    return model


def get_lstm_model():
    model = Sequential()
    model.add(LSTM(128, input_shape=(28, 28), activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
    # Compile model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'],
    )
    return model

