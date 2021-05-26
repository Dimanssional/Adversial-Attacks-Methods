import tensorflow as tf


class AdversarialAttack:

    def __init__(self, eps, model):

        self.eps = eps
        self._model = model

        self.losses = [tf.compat.v1.losses.softmax_cross_entropy, tf.keras.losses.MSE]

        self.name = None

    def adversarial_pattern(self, input_image, **kwargs):
        pass
