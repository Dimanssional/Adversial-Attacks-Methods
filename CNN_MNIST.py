import sys
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam


class CNNmodel(object):

    def __init__(self, img_x, img_y, channels, num_classes, name, epochs=1):

        self.img_x = img_x
        self.img_y = img_y

        self.channels = channels
        self.num_classes = num_classes

        self.name = name

        self.epochs = epochs
        self.opt = Adam(lr=0.001)

    def model(self):
        model = Sequential(name=self.name)

        model.add(Conv2D(32, kernel_size=(3, 3), strides=(3, 3), padding='same', activation='relu',
                         input_shape=(self.img_x, self.img_y, self.channels)))
        model.add(Conv2D(64, kernel_size=(3, 3), strides=(3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, kernel_size=(3, 3), strides=(3, 3), padding='same', activation='relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(32))
        model.add(Dropout(0.2))
        model.add(Dense(32))
        model.add(Dropout(0.2))

        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(optimizer=self.opt, loss='mse', metrics=['accuracy'])

        return model

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

    def evaluate(self, X, y, model):
        evaluation = model.evaluate(X, y)
        return evaluation

    @staticmethod
    def get_logits(model):
        model.layers[-1].activation = None


if __name__ == '__main__':

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    img_cols, img_rows, channels = 28, 28, 1
    num_classes = 10

    X_train = X_train / 255
    X_test = X_test / 255

    X_train = X_train.reshape((-1, img_rows, img_cols, channels))
    X_test = X_test.reshape((-1, img_rows, img_cols, channels))

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    CNN = CNNmodel(img_rows, img_cols, channels, num_classes, "Convolutional_Neural_Network_trained_on_MNIST")
    CNN_model = CNN.model()

    history = CNN_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    evaluation = CNN.evaluate(X_test, y_test, CNN_model)
    CNN_model.summarize_diagnostics(history)

    print("\n%s: %.2f%%" % (CNN_model.metrics_names[1], evaluation[1] * 100))

    ask = input("Press S to save model to drive...")

    if ask.upper() == "S":
        saved_model = CNN_model.to_json()
        with open("CNN_model_mnist.json", "w") as json_file:
            json_file.write(saved_model)
        CNN_model.save_weights('CNN_model_mnist.h5')


