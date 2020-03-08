import os
from typing import Dict

import numpy as np
from PIL import Image
from keras import layers, models
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

PATH = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(PATH, 'asl_alphabet_train/asl_alphabet_train/')


def create_lookup_dict(path: str, reverse: bool) -> Dict[str, int]:
    lookup_dict = {}
    for c, f in enumerate(os.listdir(path)):
        if not f.startswith('.') and f != 'desktop.ini':
            if not reverse:
                lookup_dict[f] = c
            else:
                lookup_dict[c] = f
    return lookup_dict


def read_images() -> tuple:
    # Read in images
    x_data = []
    y_data = []
    data_count = 0  # We'll use this to tally how many images are in our dataset
    for sub_folder in os.listdir(TRAIN):
        count = 0  # To tally images of a given gesture
        temp = os.path.join(TRAIN, sub_folder + '/')
        for i in range(100):
            image = os.listdir(temp)[i]
            img = Image.open(os.path.join(temp, image)).convert('L')
                        # Read in and convert to greyscale
            img = img.resize((320, 120))
            arr = np.array(img)
            x_data.append(arr)
            count = count + 1
        y_values = np.full((count, 1), lookup[sub_folder])
        y_data.append(y_values)
        data_count = data_count + count
    x_data = np.array(x_data, dtype='float32')
    y_data = np.array(y_data)
    y_data = y_data.reshape(data_count, 1)  # Reshape to be the correct size

    return x_data, y_data, data_count


if __name__ == '__main__':
    # A dictionary storing the names of gestures with numerical identifiers
    lookup = create_lookup_dict(TRAIN, False)

    # Which gesture is associated to a given identifier
    reverse_lookup = create_lookup_dict(TRAIN, True)

    x_data, y_data, data_count = read_images()

    y_data = to_categorical(y_data)
    x_data = x_data.reshape((data_count, 120, 320, 1))
    x_data /= 255

    x_train, x_further, y_train, y_further = train_test_split(x_data, y_data,
                                                              test_size=0.2)
    x_validate, x_test, y_validate, y_test = train_test_split(x_further,
                                                              y_further,
                                                              test_size=0.5)


    model = models.load_model(os.path.join(PATH, 'asl_model.h5'))
    model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=1,
              validation_data=(x_validate, y_validate))

    [loss, acc] = model.evaluate(x_test, y_test, verbose=1)

    model.save(os.path.join(PATH, 'asl_model.h5'))
