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



################## Alvincliu's Pairwise and Listwise Loss ##################
################## Pairwise Loss ##################
def BPR_loss(pos, neg):
    '''
    :param pos: the prediction of the positive item (tf.Tensor)
    :param neg: the prediction of the negative item (tf.Tensor)
    Description:
    Return the BPR loss
    '''
    return -tf.reduce_mean(tf.math.log(tf.nn.sigmoid(pos - neg)+1e-8))

def SAUC_loss(pos, neg, tau=0.5):
    '''
    :param pos: the prediction of the positive item (tf.Tensor)
    :param neg: the prediction of the negative item (tf.Tensor)
    Description:
    Return the SAUC loss
    '''
    return -tf.reduce_mean(tf.math.log(tf.nn.sigmoid((pos-neg)/tau)+1e-8))

################## Listwise Loss ##################
def ListCE_loss(label_list, pred, N, alpha=0.5):
    # sigmoid pred
    pred = tf.nn.sigmoid(pred) + 1e-8
    sigmoid_loss = tf.keras.losses.BinaryCrossentropy()(label_list, pred)
    # group the data by userid, i.e., groupby N columns, and then reshape
    grouped_pred = tf.reshape(pred, [-1, N])
    # group the label by userid, i.e., groupby N columns
    grouped_label = tf.reshape(label_list, [-1, N])
    list_loss = -tf.reduce_mean(tf.reduce_sum(grouped_label * tf.math.log(grouped_pred/tf.reduce_sum(grouped_pred, axis=1, keepdims=True)), axis=1))
    # total loss
    loss = alpha * list_loss + (1 - alpha) * sigmoid_loss
    return tf.reduce_mean(loss)

def SC_loss(label_list, pred, N, y0):
    """
    :param label_list: label list
    :param pred: prediction
    :param N: number of items
    :param y0:  global tunable parameter as label, where the score is fixed to 0
    :return: loss
    """
    # group the data by userid, i.e., groupby N columns, and then reshape
    grouped_pred = tf.reshape(pred, [-1, N])
    # group the label by userid, i.e., groupby N columns
    grouped_label = tf.reshape(label_list, [-1, N])

    # create a tensor full of y0, shape is (batch_size/N, 1)
    reference_label = tf.reshape(tf.repeat(y0, repeats=grouped_label.shape[0]), [-1, 1])
    # convert the label to float
    grouped_label = tf.cast(grouped_label, tf.float32)
    # concatenate the reference label and the grouped label
    grouped_label = tf.concat([reference_label, grouped_label], axis=1)
    grouped_label = tf.nn.softmax(grouped_label, axis=1)

    # create a reference prediction, shape is (batch_size/N, 1) and full of 0.0
    reference_pred = tf.reshape(tf.zeros_like(reference_label), [-1, 1])
    # concatenate the reference prediction and the grouped prediction
    grouped_pred = tf.concat([reference_pred, grouped_pred], axis=1)

    # calculate the sotfmax loss
    softmax_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(grouped_label, grouped_pred)
    return tf.reduce_mean(softmax_loss)

def JRC_loss(label_list, pos_logits, neg_logits, N, alpha=0.1):
    '''
    :param label_list: the label list
    :param pos_logits: the positive logits
    :param neg_logits: the negative logits
    :param N: the number of items per user
    :param alpha: the weight of calibration loss
    :return: the loss
    '''
    pos = tf.nn.sigmoid(pos_logits-neg_logits)
    calib_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(label_list, pos)

    # group the data by userid, i.e., groupby N columns
    grouped_pos_logits = tf.reshape(pos_logits, [-1, N])
    grouped_neg_logits = tf.reshape(neg_logits, [-1, N])
    # group the label by userid, i.e., groupby N columns
    grouped_label = tf.reshape(label_list, [-1, N])

    # calculate the ranking loss
    ranking_loss_pos = - tf.reduce_mean(tf.math.log(tf.exp(grouped_pos_logits)*grouped_label/tf.reduce_sum(tf.exp(grouped_pos_logits) + 1e-8, axis=1, keepdims=True)))

    # reverse the label
    neg_grouped_label = 1 - grouped_label
    ranking_loss_neg = - tf.reduce_mean(tf.math.log(tf.exp(grouped_neg_logits)*neg_grouped_label/tf.reduce_sum(tf.exp(grouped_neg_logits) + 1e-8, axis=1, keepdims=True)))
    ranking_loss = ranking_loss_pos + ranking_loss_neg

    # total loss
    loss = alpha * calib_loss + (1 - alpha) * ranking_loss
    return loss

def rank_loss(pre, rank_label):
    reduce_matrix = pre - tf.transpose(pre)
    sign_matrix = tf.sign(rank_label - tf.transpose(rank_label))
    sign_matrix = tf.cast(sign_matrix, dtype="float")
    return -K.mean(K.log(K.sigmoid(sign_matrix * reduce_matrix)),axis=-1)

