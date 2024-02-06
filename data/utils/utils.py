# -*- coding:utf-8 -*-

import random
import numpy as np
import pandas as pd
import tensorflow as tf
from itertools import combinations

def generate_pairs(ranges=range(1, 100), mask=None, order=2):
    res = []
    for i in range(order):
        res.append([])
    for i, pair in enumerate(list(combinations(ranges, order))):
        if mask is None or mask[i]==1:
            for j in range(order):
                res[j].append(pair[j])
    print("generated pairs", len(res[0]))
    return res

def df_to_dict(df: pd.DataFrame, cols: str):
    """dataframe to dict"""

    key = df[cols[0]].values
    val = df[cols[1]].values
    return dict(zip(key, val))


def split_train_test_by_id(data_size, src_id, ratio=0.8, seed=2021):
    """split the dataset in a way that the training and testing sets have distinct src_id.

    :param data_size: size of the entire dataset
    :param src_id: list of the player id
    :param ratio: ratio of the training set
    :param seed: random seed
    """
    np.random.seed(seed)
    train_list = []
    test_list = []
    distinct_src_id = list(set(src_id))
    random.shuffle(distinct_src_id)
    src_id_size = len(distinct_src_id)
    test_id = set(distinct_src_id[int(src_id_size*ratio):])
    for i in range(data_size):
        if src_id[i] in test_id :
            test_list.append(i)
        else:
            train_list.append(i)
    random.shuffle(train_list)
    random.shuffle(test_list)
    return train_list, test_list


def split_train_test(data_size, ratio = 0.8, seed = 2021):
    """randomly split the dataset into training and testing set"""

    np.random.seed(seed)
    random_list = np.random.permutation(data_size)
    return random_list[:int(data_size * ratio)], random_list[int(data_size * ratio):]


def concat_func(inputs, axis=-1, mask=False):
    if not mask:
        inputs = list(map(NoMask(), inputs))
    if len(inputs) == 1:
        return inputs[0]
    else:
        return tf.keras.layers.Concatenate(axis=axis)(inputs)


def reduce_mean(input_tensor,
                axis=None,
                keep_dims=False,
                name=None,
                reduction_indices=None):
    try:
        return tf.reduce_mean(input_tensor,
                              axis=axis,
                              keep_dims=keep_dims,
                              name=name,
                              reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_mean(input_tensor,
                              axis=axis,
                              keepdims=keep_dims,
                              name=name)


def reduce_sum(input_tensor,
               axis=None,
               keep_dims=False,
               name=None,
               reduction_indices=None):
    try:
        return tf.reduce_sum(input_tensor,
                             axis=axis,
                             keep_dims=keep_dims,
                             name=name,
                             reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_sum(input_tensor,
                             axis=axis,
                             keepdims=keep_dims,
                             name=name)


def reduce_max(input_tensor,
               axis=None,
               keep_dims=False,
               name=None,
               reduction_indices=None):
    try:
        return tf.reduce_max(input_tensor,
                             axis=axis,
                             keep_dims=keep_dims,
                             name=name,
                             reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_max(input_tensor,
                             axis=axis,
                             keepdims=keep_dims,
                             name=name)


class NoMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NoMask, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(NoMask, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        return x

    def compute_mask(self, inputs, mask):
        return None
