import cv2
import numpy as np
import os
from keras import layers, models
from keras.models import load_model
import h5py
import random
from statistics import mode, StatisticsError

PATH = os.path.dirname(os.path.abspath(__file__))
TESTING = os.path.join(PATH, 'Testing')

LABELS = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z'.split()


def load_model() -> models.Sequential():
    global model
    model = models.load_model(os.path.join(PATH, 'asl_model.h5'))

    return model

def predict_image(image, modeli):
    pred_array = modeli.predict(image)
    m = []
    for i in pred_array:
        m.append(np.argmax(i))
    # model.predict() returns an array of probabilities -
    # np.argmax grabs the index of the highest probability.
    for _ in range(len(m)):
        try:
            a = mode(m)
        except StatisticsError:
            m.pop()
            a = 'error'
        if a is int:
            break

    result = LABELS[a]

    # A bit of magic here - the score is a float, but I wanted to
    # display just 2 digits beyond the decimal point.
    return result

def main():
    modeli = load_model()

    cap = cv2.VideoCapture(0)

    while True:

        _, frame = cap.read()
        frame = cv2.resize(frame, (800, 600))
        cv2.imshow('frame', frame)
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        k = cv2.waitKey(1)
        if k == 32:
            grey = np.reshape(grey, (-1, 200, 200, 1))
            ans = predict_image(grey, modeli)
            print(ans)

        if cv2.waitKey(1) & 0xFF == ord('p'):
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
