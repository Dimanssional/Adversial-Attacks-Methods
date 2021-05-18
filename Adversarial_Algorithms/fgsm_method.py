import tensorflow as tf
import numpy as np

from adversarial_interface import AdversarialAttack


class FGSM(AdversarialAttack):

    def __init__(self, eps, model):
        super().__init__(eps, model)

        self.name = "Fast-Gradient Sign Method"

    def adversarial_pattern(self, input_image, **kwargs):
        input_image = tf.cast(input_image, tf.float32)
        true_label = kwargs.get("true_label")
        print(true_label.shape)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(input_image)
            probs = self._model(input_image)
            loss = self.losses[1](true_label, probs)

        gradients = tape.gradient(loss, input_image)
        perturbations = self.eps * tf.sign(gradients)

        adv_img = input_image + perturbations

        return adv_img, np.array(perturbations)
