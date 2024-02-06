# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, Dense, Dropout, BatchNormalization


class attention_dense_layer_v3(Layer):
    def __init__(self, activation='sigmoid'):
        super(attention_dense_layer_v3, self).__init__()
        self.activation = activation
        self.layer_1st = Dense(16, activation="relu")
        self.layer_2nd = Dense(1, activation=None)

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.layer_1st(x)
        x = self.layer_2nd(x)
        return x


class attention_dense_layer_v2(Layer):
    def __init__(self, activation='sigmoid'):
        super(attention_dense_layer_v2, self).__init__()
        self.activation = activation
        self.layer_1st = Dense(16, activation="relu")
        self.layer_sigmoid = Dense(1, activation="sigmoid")
        self.layer_tanh = Dense(1, activation="tanh")
        self.layer_relu = Dense(1, activation="relu")
        self.bn = BatchNormalization()
    def call(self, inputs, **kwargs):
        x = inputs
        x = self.layer_1st(x)
        if self.activation=="sigmoid":
            x = self.layer_sigmoid(x)
        elif self.activation=="mask_relu":
            x = self.bn(x)
            x = self.layer_tanh(x)
            x = tf.nn.relu(x)
        elif self.activation=="relu":
            x = self.layer_relu(x)
        elif self.activation=="tanh":
            x = self.layer_tanh(x)
        elif self.activation == "no_mask":
            x = self.layer_sigmoid(x)
            x = x/x
        return x


class attention_dense_layer(Layer):
    def __init__(self, activation='sigmoid'):
        super(attention_dense_layer, self).__init__()
        self.activation = activation
        self.layer_1st = Dense(16, activation="relu")
        self.layer_sigmoid = Dense(1, activation="sigmoid")
        self.layer_tanh = Dense(1, activation="tanh")
        self.layer_relu = Dense(1, activation="relu")
    def call(self, inputs, **kwargs):
        x = inputs
        x = self.layer_1st(x)
        if self.activation=="sigmoid":
            x = self.layer_sigmoid(x)
        elif self.activation=="mask_relu":
            x = self.layer_tanh(x)
            x = tf.nn.relu(x)
        elif self.activation=="relu":
            x = self.layer_relu(x)
        elif self.activation=="tanh":
            x = self.layer_tanh(x)
        elif self.activation == "no_mask":
            x = self.layer_sigmoid(x)
            x = x/x
        return x

class DenseLayer(Layer):
    """Dense Layer
    """
    def __init__(self, hidden_units, activation='relu', dropout=0.0):
        super(DenseLayer, self).__init__()
        self.dense_layer = [Dense(i, activation=activation) for i in hidden_units]
        # self.dropout = Dropout(dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.dense_layer:
            x = layer(x)
        return x


class DotProductAttention(Layer):
    """Dot-Production Operation for Attention
    """
    def __init__(self, dropout=0.0):
        super(DotProductAttention, self).__init__()
        self._dropout = dropout
        self._masking_num = -2**32 + 1

    def call(self, inputs):
        # queries: [None, n, k]
        # keys:    [None, n, k]
        # values:  [None, n, k]
        queries, keys, values = inputs
        score = K.batch_dot(queries, tf.transpose(keys, [0, 2, 1]))  # [None, n, n]
        score = score/int(queries.shape[-1])**0.5   # 缩放
        score = K.softmax(score)                    # SoftMax
        score = K.dropout(score, self._dropout)     # dropout
        outputs = K.batch_dot(score, values)        # [None, n, k]
        return outputs


class MultiHeadAttention(Layer):
    """Multi-Head Attention Layer
    """
    def __init__(self, n_heads=2, head_dim=4, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self._n_heads = n_heads
        self._head_dim = head_dim
        self._dropout = dropout
        self._att_layer = DotProductAttention(dropout=self._dropout)

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        self._weights_queries = self.add_weight(
            shape=(input_shape[0][-1], self._n_heads*self._head_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='weights_queries')
        self._weights_keys = self.add_weight(
            shape=(input_shape[1][-1], self._n_heads*self._head_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='weights_keys')
        self._weights_values = self.add_weight(
            shape=(input_shape[2][-1], self._n_heads*self._head_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='weights_values')

    def call(self, inputs):
        # queries: [None, n, k]
        # keys:    [None, n, k]
        # values:  [None, n, k]
        queries, keys, values = inputs
        if self._n_heads*self._head_dim != queries.shape[-1]:
            raise ValueError("n_head * head_dim not equal embedding dim {}".format(queries.shape[-1]))

        queries_linear = K.dot(queries, self._weights_queries)  # [None, n, k]
        keys_linear = K.dot(keys, self._weights_keys)           # [None, n, k]
        values_linear = K.dot(values, self._weights_values)     # [None, n, k]

        queries_multi_heads = tf.concat(tf.split(queries_linear, self._n_heads, axis=2),
                                        axis=0)  # [None*n_head, n, k/n_head]
        keys_multi_heads = tf.concat(tf.split(keys_linear, self._n_heads, axis=2),
                                     axis=0)        # [None*n_head, n, k/n_head]
        values_multi_heads = tf.concat(tf.split(values_linear, self._n_heads, axis=2),
                                       axis=0)    # [None*n_head, n, k/n_head]

        att_out = self._att_layer([queries_multi_heads, keys_multi_heads, values_multi_heads])   # [None*n_head, n, k/n_head]
        outputs = tf.concat(tf.split(att_out, self._n_heads, axis=0), axis=2)    # [None, n, k]
        return outputs


class DotProductAttention_mask(Layer):
    """Dot-Production Operation for Attention
    """
    def __init__(self, dropout=0.0):
        super(DotProductAttention_mask, self).__init__()
        self._dropout = dropout
        self._masking_num = -2**32 + 1

    def call(self, inputs):
        # queries: [None, n, k]
        # keys:    [None, n, k]
        # values:  [None, n, k]
        queries, keys, values = inputs
        score = K.batch_dot(queries, tf.transpose(keys, [0, 2, 1]))  # [None, n, n]
        score = score/int(queries.shape[-1])**0.5   # 缩放
        score = tf.nn.tanh(score)
        score = tf.nn.relu(score)
        score = K.dropout(score, self._dropout)     # dropout
        outputs = K.batch_dot(score, values)        # [None, n, k]
        return outputs


class MultiHeadAttention_mask(Layer):
    """Multi-Head Attention Layer
    """
    def __init__(self, n_heads=2, head_dim=4, dropout=0.1):
        super(MultiHeadAttention_mask, self).__init__()
        self._n_heads = n_heads
        self._head_dim = head_dim
        self._dropout = dropout
        self._att_layer = DotProductAttention_mask(dropout=self._dropout)

    def build(self, input_shape):
        super(MultiHeadAttention_mask, self).build(input_shape)
        self._weights_queries = self.add_weight(
            shape=(input_shape[0][-1], self._n_heads*self._head_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='weights_queries')
        self._weights_keys = self.add_weight(
            shape=(input_shape[1][-1], self._n_heads*self._head_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='weights_keys')
        self._weights_values = self.add_weight(
            shape=(input_shape[2][-1], self._n_heads*self._head_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='weights_values')

    def call(self, inputs):
        # queries: [None, n, k]
        # keys:    [None, n, k]
        # values:  [None, n, k]
        queries, keys, values = inputs
        if self._n_heads*self._head_dim != queries.shape[-1]:
            raise ValueError("n_head * head_dim not equal embedding dim {}".format(queries.shape[-1]))

        queries_linear = K.dot(queries, self._weights_queries)  # [None, n, k]
        keys_linear = K.dot(keys, self._weights_keys)           # [None, n, k]
        values_linear = K.dot(values, self._weights_values)     # [None, n, k]

        queries_multi_heads = tf.concat(tf.split(queries_linear, self._n_heads, axis=2),
                                        axis=0)  # [None*n_head, n, k/n_head]
        keys_multi_heads = tf.concat(tf.split(keys_linear, self._n_heads, axis=2),
                                     axis=0)        # [None*n_head, n, k/n_head]
        values_multi_heads = tf.concat(tf.split(values_linear, self._n_heads, axis=2),
                                       axis=0)    # [None*n_head, n, k/n_head]

        att_out = self._att_layer([queries_multi_heads, keys_multi_heads, values_multi_heads])   # [None*n_head, n, k/n_head]
        outputs = tf.concat(tf.split(att_out, self._n_heads, axis=0), axis=2)    # [None, n, k]
        return outputs