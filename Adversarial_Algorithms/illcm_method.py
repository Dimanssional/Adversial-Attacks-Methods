import tensorflow as tf
import numpy as np

from adversarial_interface import AdversarialAttack


class ILLCM(AdversarialAttack):

    def __init__(self, eps, model, clip_min=0.0, clip_max=1.0):
        super().__init__(eps, model)

        self.clip_min = clip_min
        self.clip_max = clip_max

        self.name = "Iterative Least-Likely Class Method"

    def adversarial_pattern(self, input_image, **kwargs):
        input_image = tf.cast(input_image, tf.float32)
        iterations, alpha = kwargs.get("iterations"), kwargs.get("alpha")

        probs = self._model(input_image)
        Y_ll = tf.argmin(probs, 1)

        for _ in tf.range(iterations):

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(input_image)
                probs = self._model(input_image)
                loss = self.losses[1](Y_ll, probs)

            gradients = tape.gradient(loss, input_image)
            perturbations = alpha * tf.sign(gradients)

            adv_img = input_image - perturbations

            adv_img = tf.clip_by_value(adv_img, input_image - self.eps, input_image + self.eps)
            adv_img = tf.clip_by_value(adv_img, self.clip_min, self.clip_max)

        return adv_img, np.int(Y_ll)
