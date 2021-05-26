import tensorflow as tf
import numpy as np

from adversarial_interface import AdversarialAttack


class BIM(AdversarialAttack):

    def __init__(self, eps, model, clip_min=0.0, clip_max=1.0):
        super().__init__(eps, model)

        self.clip_min = clip_min
        self.clip_max = clip_max

        self.name = "Basic Iterative Method"

    def __get_logits(self):
        self._model.layers[-1].activation = None

    def adversarial_pattern(self, input_image, **kwargs):
        input_image = tf.cast(input_image, tf.float32)
        true_label, iterations, alpha = kwargs.get("true_label"), kwargs.get("iterations"), kwargs.get("alpha")

        self.__get_logits()
        adv_img = input_image
        for _ in tf.range(iterations):

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(adv_img)
                logits = self._model(adv_img)
                loss = self.losses[0](true_label, tf.reshape(logits, (-1, )))

            gradients = tape.gradient(loss, adv_img)
            perturbations = alpha * tf.sign(gradients)

            adv_img = adv_img + perturbations

            adv_img = tf.clip_by_value(adv_img, input_image - self.eps, input_image + self.eps)
            adv_img = tf.clip_by_value(adv_img, self.clip_min, self.clip_max)

        return adv_img, np.array(perturbations)
