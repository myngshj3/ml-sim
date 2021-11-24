import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

def main():
    test_data = np.array(([0,0],[0,1],[1,0],[1,1]), dtype=np.float32)
    model = keras.models.load_model(os.path.join('result', 'outmodel'))
    predictions = model.predict(test_data)
    print(predictions)
    for i, prediction in enumerate(predictions):
        result = np.argmax(prediction)
        print(f'input: {test_data[i]}, result: {result}')


if __name__ == "__main__":
    main()

