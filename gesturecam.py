import cv2
import numpy as np
import os
from keras import layers, models
from keras.models import load_model
import h5py


PATH = os.path.dirname(os.path.abspath(__file__))
TESTING = os.path.join(PATH, 'Testing')

LABELS = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z'.split()


def load_model() -> models.Sequential():
    # model = models.Sequential()
    # model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu',
    #                         input_shape=(120, 320, 1)))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(128, activation='relu'))
    # model.add(layers.Dense(10, activation='softmax'))

    # model.load_weights(os.path.join(PATH, 'sample_model.h5'))
    global model
    model = models.load_model(os.path.join(PATH, 'sample_model.h5'))

    return model


def predict_image(image, modeli):
    pred_array = modeli.predict(image)

    # model.predict() returns an array of probabilities -
    # np.argmax grabs the index of the highest probability.
    result = LABELS[np.argmax(pred_array)]

    # A bit of magic here - the score is a float, but I wanted to
    # display just 2 digits beyond the decimal point.
    return result

def main():
    cap = cv2.VideoCapture(0)

    modeli = load_model()
    modeli.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    while True:

        _, frame = cap.read()
        cv2.imshow('frame', frame)

        k = cv2.waitKey(1)
        if k == 32:
            fixed_frame = np.reshape(frame, (24, 120, 320, 1))
            ans = predict_image(fixed_frame, modeli)
            print(ans)

        if cv2.waitKey(1) & 0xFF == ord('p'):
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
