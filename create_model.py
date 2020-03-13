import os
from typing import Dict
import numpy as np
from PIL import Image
from keras import layers, models
from sklearn.model_selection import train_test_split
import random

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
    x_data = []
    y_data = []
    data_count = 0  # We'll use this to tally how many images are in our dataset
    for sub_folder in os.listdir(TRAIN):
        count = 0  # To tally images of a given gesture
        temp = os.path.join(TRAIN, sub_folder + '/')

        # As we were using a large dataset, this helped keep pre-processing
        # times down
        random.seed()
        l = random.randint(0, 2850)
        for i in range(l, l+150):
            image = os.listdir(temp)[i]

            #Convert images to greyscale
            img = Image.open(os.path.join(temp, image)).convert('L')

            arr = np.array(img)
            x_data.append(arr)
            count += 1
        y_values = np.full((count, 1), lookup[sub_folder])
        y_data.extend(y_values)
        data_count = data_count + count
    x_data = np.array(x_data, dtype='float32')
    y_data = np.array(y_data)

    return x_data, y_data, data_count


def form_nn() -> models.Sequential(): # Only call this if model is not yet made

    model = models.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), activation='relu',
                            input_shape=(200, 200, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(26, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == '__main__':
    # A dictionary storing the names of gestures with numerical identifiers
    lookup = create_lookup_dict(TRAIN, False)

    # Which gesture is associated to a given identifier
    reverse_lookup = create_lookup_dict(TRAIN, True)

    x_data, y_data, data_count = read_images()

    # Reshape for the Conv2D input layer and standardize RGB values, since we
    # are using greyscale images
    x_data = x_data.reshape((data_count, 200, 200, 1))
    x_data /= 255

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)

    model = models.load_model(os.path.join(PATH, 'asl_model.h5'))

    # model = form_nn()

    model.fit(x=x_train, y=y_train, epochs=5, batch_size=50, verbose=1)

    [loss, acc] = model.evaluate(x_test, y_test, verbose=1)
    print(loss, acc)

    model.save(os.path.join(PATH, 'asl_model.h5'), overwrite=True)
