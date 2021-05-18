import tensorflow as tf
import numpy as np

from adversarial_interface import AdversarialAttack


class JSMA(AdversarialAttack):

    def __init__(self, X, Y, channels, eps, model, clip_min=0.0, clip_max=1.0):
        super().__init__(eps, model)

        self.img_rows, self.img_cols = X, Y
        self.channels = channels

        self.clip_min = clip_min
        self.clip_max = clip_max

        self.name = "Jacobian-based Saliecny Map Attack"

    def __get_logits(self):
        self._model.layers[-1].activation = None

    def __compute_jacobian_matrix(self, input_image, target_image, y_inidx, y_tidx):
        self.__get_logits()

        logits = self._model(input_image)

        logits = np.array(tf.reshape(logits, (-1,)))
        logidx, logidx_target = logits[y_inidx], logits[y_tidx]

        jacobian, jacobian_target = np.gradient(input_image.ravel(), logidx), \
                                    np.gradient(target_image.ravel(), logidx_target)

        return tf.cast(jacobian.reshape((1, self.img_cols, self.img_rows, self.channels)), tf.float32), \
               tf.cast(jacobian_target.reshape((1, self.img_cols, self.img_rows, self.channels)), tf.float32)

    def __compute_saliency_map(self, input_image, dF_tX, dF_jX):
        # compute saliency map with respect to jacobian
        c1, c2 = tf.logical_or(self.eps < 0, input_image < self.clip_max), \
                 tf.logical_or(self.eps > 0, input_image > self.clip_min)

        sal1 = dF_tX >= 0
        sal2 = dF_jX <= 0

        shape = input_image.ravel().shape[0]

        condition = tf.cast(tf.reduce_all([c1, c2, sal1, sal2], axis=0), dtype=tf.float32)
        score_sm = condition * (dF_tX * tf.abs(dF_jX))

        score_sm = tf.reshape(score_sm, shape=[1, shape])
        return score_sm

    def adversarial_pattern(self, input_image, **kwargs):
        # generate adversarial perturbations
        max_iter, target_image, y_inidx, y_tidx = kwargs.get("max_iter"), kwargs.get("target_image"),\
                                                  kwargs.get("y_inidx"), kwargs.get("y_tidx")
        perturbations = []

        jacobian_input, jacobian_target = self.__compute_jacobian_matrix(input_image, target_image, y_inidx, y_tidx)
        jacobian_adv = jacobian_input - jacobian_target

        saliency_score = self.__compute_saliency_map(input_image, jacobian_target, jacobian_adv)
        shape = input_image.ravel().shape[0]
        update_map = np.ones((1, shape), dtype=np.float32)

        i = 0
        while i < max_iter:
            # self.eps = np.random.uniform(0.01, 2.5)
            idx = tf.argmax(saliency_score, axis=1)
            update_map[:, int(idx)] = 0.0
            update_map = tf.constant(update_map, dtype=tf.float32)
            saliency_score = saliency_score * update_map

            update_map = np.array(update_map, dtype=np.float32)
            perturbation = tf.one_hot(idx, shape, on_value=self.eps, off_value=0.0)
            perturbation = tf.reshape(perturbation, shape=tf.shape(input_image))

            perturbations.append(perturbation)
            i += 1

        pert_sum = tf.zeros((1, self.img_cols, self.img_rows, self.channels), dtype=tf.float32)
        for pert in perturbations:
            pert_sum += pert

        adversarial_image = tf.clip_by_value(tf.stop_gradient(input_image + pert_sum), self.clip_min, self.clip_max)

        return np.array(adversarial_image), np.array(pert_sum)
