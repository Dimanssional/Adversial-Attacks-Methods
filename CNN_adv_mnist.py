import sys

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

sys.path.insert(1, "Adversarial_Algorithms")
from fgsm_method import FGSM
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import model_from_json
import random

import tensorflow as tf

jfile = open("CNN_model_mnist.json", "r")
loaded_json = jfile.read()
jfile.close()

loaded_model = model_from_json(loaded_json)
loaded_model.load_weights("CNN_model_mnist.h5")
loaded_model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])

loaded_model.summary()

(X_train, y_train), (X_test, y_test) = mnist.load_data()

img_cols, img_rows, channels = 28, 28, 1

num_classes = 10

X_train = X_train / 255
X_test = X_test / 255

X_train = X_train.reshape((-1, img_rows, img_cols, channels))
X_test = X_test.reshape((-1, img_rows, img_cols, channels))

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)


def generate_adversarials(batch_size):
    while True:
        x = []
        y = []
        for batch in range(batch_size):
         # if batch_size > 10000 and batch % 1000 == 0:
            #     print(batch/batch_size)

            N = random.randint(0, 100)

            label = y_train[N]
            image = X_train[N]

            fgsm = FGSM(0.1, loaded_model)
            adversarial, _ = fgsm.adversarial_pattern(image.reshape((1, img_rows, img_cols, channels)),
                                                  true_label=label)

            x.append(adversarial)
            y.append(y_train[N])

        x = np.asarray(x).reshape((batch_size, img_rows, img_cols, channels))
        y = np.asarray(y)

        yield x, y


if __name__ == '__main__':

    x_adversarial_train, y_adversarial_train = next(generate_adversarials(20000))
    x_adversarial_test, y_adversarial_test = next(generate_adversarials(10000))

    print(x_adversarial_train.shape, y_adversarial_train.shape, "\n")
    print(x_adversarial_test.shape, y_adversarial_test.shape)

    CNN_model_adversarial_trained = loaded_model

    CNN_model_adversarial_trained.fit(x_adversarial_train, y_adversarial_train, epochs=15, batch_size=32,
                                      validation_data=(X_test, y_test))

    eval_with_adversarial = CNN_model_adversarial_trained.evaluate(
        x_adversarial_test, y_adversarial_test)

    eval_regular = CNN_model_adversarial_trained.evaluate(X_test, y_test)

    print("Defended %s on adversarial images: %.2f%%" % (
    CNN_model_adversarial_trained.metrics_names[1], eval_with_adversarial[1] * 100))
    print("Defended %s on regular images: %.2f%%" % (
    CNN_model_adversarial_trained.metrics_names[1], eval_regular[1] * 100))

    x_adversarial_test1, y_adversarial_test1 = next(generate_adversarials(10000))
    eval_with_adversarial = CNN_model_adversarial_trained.evaluate(
        x_adversarial_test1, y_adversarial_test1)
    print("Defended %s on adversarial images: %.2f%%" % (
    CNN_model_adversarial_trained.metrics_names[1], eval_with_adversarial[1] * 100))

    ask = input("Press S to save model to drive...")

    if ask.upper() == "S":
        saved_model = CNN_model_adversarial_trained.to_json()
        with open("CNN_model_mnist_adversarial.json", "w") as json_file:
            json_file.write(saved_model)
        CNN_model_adversarial_trained.save_weights('CNN_model_mnist_adversarial.h5')





