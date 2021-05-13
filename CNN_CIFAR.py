import sys
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os

# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
# from keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D, BatchNormalization
# from keras.models import Sequential
# from keras.optimizers import SGD


class CNNmodel2(object):

    def __init__(self, img_x, img_y, channels, num_classes, name, epochs=1):

        self.img_x = img_x
        self.img_y = img_y

        self.channels = channels
        self.num_classes = num_classes

        self.name = name

        self.epochs = epochs
        self.opt = SGD(lr=0.001, momentum=0.9)

    def model(self):

        model = Sequential(name=self.name)
        model.add(Conv2D(32, kernel_size=(3, 3), kernel_initializer="he_uniform",
                         activation="relu", padding="same", input_shape=(self.img_x, self.img_y, self.channels)))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=(3, 3), kernel_initializer="he_uniform", activation="relu", padding="same"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, kernel_size=(3, 3), kernel_initializer="he_uniform",
                         activation="relu", padding="same"))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3, 3), kernel_initializer="he_uniform",
                         activation="relu", padding="same"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(128, kernel_size=(3, 3), kernel_initializer="he_uniform",
                         activation="relu", padding="same"))
        model.add(BatchNormalization())
        model.add(Conv2D(128, kernel_size=(3, 3), kernel_initializer="he_uniform",
                         activation="relu", padding="same"))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dropout(0.4))

        model.add(Dense(128, activation="relu", kernel_initializer="he_uniform"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation="softmax"))

        model.compile(optimizer=self.opt, loss="categorical_crossentropy", metrics=['accuracy'])

        return model

    def evaluate(self, X, y, model):
        evaluation = model.evaluate(X, y)
        return evaluation

    @staticmethod
    def summarize_diagnostics(hist):

        plt.subplot(221)
        plt.title("Cross Entropy Loss")
        plt.plot(hist.history['loss'], color='blue', label='train')
        plt.plot(hist.history['val_loss'], color='orange', label='test')

        plt.subplot(212)
        plt.title("Classification accuracy")
        plt.plot(hist.history['accuracy'], color='blue', label='train')
        plt.plot(hist.history['val_accuracy'], color='orange', label='test')

        path = sys.argv[0].split('/')[-1]
        plt.savefig(path + "_plot.png")
        plt.close()

    @staticmethod
    def get_logits(model):
        model.layers[-1].activation = None


if __name__ == '__main__':

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    img_cols, img_rows, channels = 32, 32, 3
    num_classes = 10

    X_train = X_train / 255
    X_test = X_test / 255

    X_train = X_train.reshape((-1, img_rows, img_cols, channels))
    X_test = X_test.reshape((-1, img_rows, img_cols, channels))

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    plt.figure(figsize=(10, 10))

    plt.subplot(221)
    plt.imshow(X_train[1].reshape(img_rows, img_cols, channels))

    plt.subplot(222)
    plt.imshow(X_test[21].reshape(img_rows, img_cols, channels))

    plt.subplot(223)
    plt.imshow(X_test[44].reshape(img_rows, img_cols, channels))

    plt.subplot(224)
    plt.imshow(X_train[54].reshape(img_rows, img_cols, channels))

    plt.show()
    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

    steps = int(X_train.shape[0] / 64)

    it_data = datagen.flow(X_train, y_train, batch_size=64)

    CNN = CNNmodel2(img_rows, img_cols, channels, num_classes, "CNN_model_cifar")
    CNN_model = CNN.model()
    history = CNN_model.fit(it_data, validation_data=(X_test, y_test), steps_per_epoch=steps, epochs=150, verbose=1)

    evaluation = CNN.evaluate(X_test, y_test, CNN_model)
    print("\n%s %.2f%%" % (CNN_model.metrics_names[1], evaluation[1] * 100))

    CNNmodel2.summarize_diagnostics(history)

    ask = input("Press S to save model to drive...")

    if ask.upper() == "S":
        saved_model = CNN_model.to_json()
        with open("CNN_model_cifar.json", "w") as json_file:
            json_file.write(saved_model)
        CNN_model.save_weights('CNN_model_cifar.h5')


