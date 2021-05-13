from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import SGD
import random

import tensorflow as tf

jfile = open("CNN_model_cifar.json", "r")
loaded_json = jfile.read()
jfile.close()

loaded_model = model_from_json(loaded_json)
loaded_model.load_weights("CNN_model_cifar.h5")
loaded_model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9), loss="categorical_crossentropy", metrics=['accuracy'])

loaded_model.summary()

(X_train_C, y_train_C), (X_test_C, y_test_C) = cifar10.load_data()

img_cols_C, img_rows_C, channels_C = 32, 32, 3
num_classes_C = 10

X_train_C = X_train_C / 255
X_test_C = X_test_C / 255

X_train_C = X_train_C.reshape((-1, img_rows_C, img_cols_C, channels_C))
X_test_C = X_test_C.reshape((-1, img_rows_C, img_cols_C, channels_C))
# print(X_train_C.shape, X_test_C.shape)

y_train_C = to_categorical(y_train_C, num_classes_C)
y_test_C = to_categorical(y_test_C, num_classes_C)

labels_C = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def adversarial_patternFGSM(image, label):
    image = tf.cast(image, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = loaded_model(image)
        loss = tf.keras.losses.MSE(label, prediction)

    gradient = tape.gradient(loss, image)
    signed_grad = tf.sign(gradient)

    return signed_grad


def generate_adversarials_C(batch_size):
    while True:
        x = []
        y = []
        for batch in range(batch_size):
         # if batch_size > 10000 and batch % 1000 == 0:
            #     print(batch/batch_size)

            N = random.randint(0, 500)

            label = y_train_C[N]
            image = X_train_C[N]

            perturbations = adversarial_patternFGSM(image.reshape((1, img_rows_C, img_cols_C, channels_C)), label).numpy()

            epsilon = 0.1
            adversarial = image + perturbations * epsilon

            x.append(adversarial)
            y.append(y_train_C[N])

        x = np.asarray(x).reshape((batch_size, img_rows_C, img_cols_C, channels_C))
        y = np.asarray(y)

        yield x, y


if __name__ == '__main__':

    x_adversarial_train, y_adversarial_train = next(generate_adversarials_C(1000))
    x_adversarial_test, y_adversarial_test = next(generate_adversarials_C(500))

    print(x_adversarial_train.shape, y_adversarial_train.shape, "\n")
    print(x_adversarial_test.shape, y_adversarial_test.shape)

    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

    steps = int(x_adversarial_train.shape[0] / 64)

    it_data = datagen.flow(x_adversarial_train, y_adversarial_train, batch_size=64)

    CNN_model_adversarial_trained = loaded_model

    CNN_model_adversarial_trained.fit(it_data, steps_per_epoch=steps, epochs=100,
                                      validation_data=(X_test_C, y_test_C))

    eval_with_adversarial = CNN_model_adversarial_trained.evaluate(
                                                x_adversarial_test, y_adversarial_test)

    eval_regular = CNN_model_adversarial_trained.evaluate(X_test_C, y_test_C)

    print("Defended %s on adversarial images: %.2f%%" % (CNN_model_adversarial_trained.metrics_names[1], eval_with_adversarial[1] * 100))
    print("Defended %s on regular images: %.2f%%" % (CNN_model_adversarial_trained.metrics_names[1], eval_regular[1] * 100))

    x_adversarial_test1, y_adversarial_test1 = next(generate_adversarials_C(10000))
    eval_with_adversarial = CNN_model_adversarial_trained.evaluate(x_adversarial_test1, y_adversarial_test1)
    print("Defended %s on adversarial images: %.2f%%" % (CNN_model_adversarial_trained.metrics_names[1], eval_with_adversarial[1] * 100))

    ask = input("Press S to save model to drive...")

    if ask.upper() == "S":
        saved_model = CNN_model_adversarial_trained.to_json()
        with open("CNN_model_cifar_adversarial.json", "w") as json_file:
            json_file.write(saved_model)
        CNN_model_adversarial_trained.save_weights('CNN_model_cifar_adversarial.h5')












