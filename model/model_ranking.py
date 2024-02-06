# -*- coding: utf-8 -*-

import itertools
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Zeros, glorot_normal
from model.layer import DenseLayer, MultiHeadAttention, attention_dense_layer, MultiHeadAttention_mask, attention_dense_layer_v2, attention_dense_layer_v3
from utils.utils import concat_func, reduce_sum, reduce_mean, generate_pairs
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import losses
# import layer normalization from keras
from tensorflow.keras.layers import LayerNormalization

class AutoInt_Vanilla(Model):
    """Vanilla AutoInt model

    params:
    num_fea_size: number of input feature dimension
    activation: activation function
    embed_dim: dimension size of the generated embedding
    dnn_dropout: dropout ratio
    n_head: number of attention heads
    attention_num: number of the attention layers
    hidden_units [num_fea_size * 8, num_fea_size * 2, num_fea_size, 64]: the list of hidden unit size
    """

    def __init__(self, num_fea_size, activation='relu', embed_dim=8,
                 dnn_dropout=0.0, n_heads=2, head_dim=4, att_dropout=0.1, attention_num=3,
                 hidden_units=[512, 256, 64]):
        super(AutoInt_Vanilla, self).__init__()
        self.num_fea_size = num_fea_size
        self.embed_dim = embed_dim
        self.dense_emb_layers = [Dense(embed_dim, activation=None) for _ in range(self.num_fea_size)]
        self.hidden_units = hidden_units
        self.attention_num = attention_num
        self.dense_layer = DenseLayer(self.hidden_units, activation, dnn_dropout)
        self.multi_head_att = MultiHeadAttention(n_heads, head_dim, att_dropout)
        self.out_layer = Dense(1, activation=None)
        # self.layer_norm = LayerNormalization()

    def call(self, x1, training=None, mask=None):
        """
        X1: dense_feature  [batch_size, dense_feature_num]
        """
        X_dense = K.concatenate([x1], axis=1)  # X_dense: [batch_size, dense_feature_num]
        dense_inputs = X_dense
        emb = [layer(tf.reshape(dense_inputs[:, i], shape=(-1, 1))) for i, layer in enumerate(self.dense_emb_layers)]
        emb = tf.transpose(emb, [1, 0, 2])
        # emb = dense_inputs

        # DNN
        dnn_input = tf.reshape(emb, shape=(-1, emb.shape[1] * emb.shape[2]))
        dnn_out = self.dense_layer(dnn_input)

        # AutoInt
        att_out = emb
        for i in range(self.attention_num):
            # att_out_res = tf.matmul(emb, self.W_res)
            att_out_res = att_out
            att_out = self.multi_head_att([att_out, att_out, att_out])
            att_out = att_out + att_out_res
            att_out = tf.nn.relu(att_out)
        att_out = tf.reshape(att_out, [-1, att_out.shape[1] * att_out.shape[2]])  # [None, 39*k]

        # output
        x = tf.concat([dnn_out, att_out], axis=-1)
        # x = dnn_out
        # x = att_out
        x = self.out_layer(x)
        # return tf.nn.sigmoid(x)
        return x



class AutoInt_JRC(Model):
    """AutoInt model for JRC loss

    params:
    num_fea_size: number of input feature dimension
    activation: activation function
    embed_dim: dimension size of the generated embedding
    dnn_dropout: dropout ratio
    n_head: number of attention heads
    attention_num: number of the attention layers
    hidden_units [num_fea_size * 8, num_fea_size * 2, num_fea_size, 64]: the list of hidden unit size
    """

    def __init__(self, num_fea_size, activation='relu', embed_dim=8,
                 dnn_dropout=0.0, n_heads=2, head_dim=4, att_dropout=0.1, attention_num=3,
                 hidden_units=[512, 256, 64]):
        super(AutoInt_JRC, self).__init__()
        self.num_fea_size = num_fea_size
        self.embed_dim = embed_dim
        self.dense_emb_layers = [Dense(embed_dim, activation=None) for _ in range(self.num_fea_size)]
        self.hidden_units = hidden_units
        self.attention_num = attention_num
        self.dense_layer = DenseLayer(self.hidden_units, activation, dnn_dropout)
        self.multi_head_att = MultiHeadAttention(n_heads, head_dim, att_dropout)
        self.out_layer = Dense(2, activation=None)
        

    def call(self, x1, training=None, mask=None):
        """
        X1: dense_feature  [batch_size, dense_feature_num]
        """
        X_dense = K.concatenate([x1], axis=1)  # X_dense: [batch_size, dense_feature_num]
        dense_inputs = X_dense
        emb = [layer(tf.reshape(dense_inputs[:, i], shape=(-1, 1))) for i, layer in enumerate(self.dense_emb_layers)]
        emb = tf.transpose(emb, [1, 0, 2])
        # emb = dense_inputs

        # DNN
        dnn_input = tf.reshape(emb, shape=(-1, emb.shape[1] * emb.shape[2]))
        dnn_out = self.dense_layer(dnn_input)

        # AutoInt
        att_out = emb
        for i in range(self.attention_num):
            att_out_res = att_out
            att_out = self.multi_head_att([att_out, att_out, att_out])
            att_out = att_out + att_out_res
            att_out = tf.nn.relu(att_out)
        att_out = tf.reshape(att_out, [-1, att_out.shape[1] * att_out.shape[2]])  # [None, 39*k]

        # output
        x = tf.concat([dnn_out, att_out], axis=-1)
        # x = dnn_out
        # x = att_out
        
        x = self.out_layer(x)
        # the first units's output is the probability of being 1
        pos_logits = tf.expand_dims(x[:, 0], axis=-1)
        # the second units's output is the probability of being 0
        neg_logits = tf.expand_dims(x[:, 1], axis=-1)
        return pos_logits, neg_logits