# -*- coding:utf-8 -*-

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import losses


def binary_focal_loss(gamma=2, alpha=0.25):
    """Binary form of focal loss.

    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)

    return binary_focal_loss_fixed


def bpr_loss(pred, label):
    """Bayesian Personalized Ranking Loss"""

    reduce_matrix = pred - tf.transpose(pred)
    sign_matrix = tf.sign(label - tf.transpose(label))
    sign_matrix = tf.cast(sign_matrix, dtype="float")
    return -K.mean(K.log(K.sigmoid(sign_matrix * reduce_matrix)),axis=-1)


def weighted_bce_loss(label, pred, pos_weight, neg_weight):
    """Weighted Binary Cross Entropy Loss"""

    neg_weight = tf.reduce_max([neg_weight, 0.1])  # ensure neg_weight will not be too small
    weights = tf.cast(tf.where(tf.equal(label, 1), pos_weight, neg_weight), tf.float32)
    bce_loss = losses.binary_crossentropy(label, pred)
    bce_loss = tf.expand_dims(bce_loss, 1)  # (None,) -> (None,1)
    return K.mean(tf.multiply(bce_loss, weights))



