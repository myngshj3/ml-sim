import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

def main():
    input_data = np.array(([0,0], [0,1], [1,0], [1,1]), dtype=np.float32)
    label_data = np.array([0, 1, 1, 1], dtype=np.int32)
    train_data, train_label = input_data, label_data
    validation_data, validation_label = input_data, label_data
    model = keras.Sequential(
        [
            keras.layers.Dense(3, activation='relu'),
            keras.layers.Dense(2, activation='softmax'),
        ]
    )
    model.compile(
        optimizer='adam', loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        x=train_data,
        y=train_label,
        epochs=1000,
        batch_size=8,
        validation_data=(validation_data, validation_label),
    )
    model.save(os.path.join('result', 'outmodel'))


if __name__ == "__main__":
    main()

