import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def main():
    digits = load_digits()
    train_data, validation_data, train_label, validation_label = train_test_split(digits.data, digits.target, test_size=0.2)
    model = keras.Sequential(
        [
            keras.layers.Dense(100, activation='relu'),
            keras.layers.Dense(100, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ]
    )
    model.compile(
        optimizer='adam', loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        x=train_data,
        y=train_label,
        epochs=20,
        batch_size=100,
        validation_data=(validation_data, validation_label),
    )
    model.save(os.path.join('result', 'outmodel'))


if __name__ == '__main__':
    main()

