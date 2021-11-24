import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from PIL import Image

def main():
    img = Image.open(os.path.join('number', '2.png'))
    img = img.convert('L')
    img = img.resize((8, 8))
    img = 16.0 - np.asarray(img, dtype=np.float32) / 16.0
    test_data = img.reshape(1, 8, 8, 1)
    model = keras.models.load_model(os.path.join('result', 'outmodel'))
    prediction = model.predict(test_data)
    result = np.argmax(prediction)
    print(f'result: {result}')


if __name__ == "__main__":
    main()

