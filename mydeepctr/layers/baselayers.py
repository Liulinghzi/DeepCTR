'''
@Author: your name
@Date: 2020-06-03 10:44:42
@LastEditTime: 2020-06-10 12:29:24
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /estimator/layers.py
'''
import tensorflow as tf


class Linear():
    # def __init__(self, dense_feature_columns, sparse_feature_columns):
    def __init__(self, l2_reg=0.0, use_bias=False):
        self.l2_reg = l2_reg
        self.use_bias = use_bias

    def __call__(self, dense_value_list, sparse_embedding_list):
        with tf.variable_scope('linear'):
            for i in range(len(dense_value_list)):
                if len(dense_value_list[i].shape) == 1:
                    dense_value_list[i] = tf.expand_dims(dense_value_list[i], axis=-1)
            if len(dense_value_list) == 0 and len(sparse_embedding_list) > 0:
                sparse_input = tf.concat(sparse_embedding_list, axis=-1)
                logits = tf.reduce_sum(sparse_input, axis=-1)
            elif len(dense_value_list) > 0 and len(sparse_embedding_list) == 0:
                dense_input = tf.concat(dense_value_list, axis=-1)
                fc = tf.layers.dense(
                        dense_input, 1, kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg), use_bias=self.use_bias)
                logits = fc
            elif len(dense_value_list) > 0 and len(sparse_embedding_list) > 0:
                dense_input = tf.concat(dense_value_list, axis=-1)
                sparse_input = tf.concat(sparse_embedding_list, axis=-1)

                fc = tf.layers.dense(
                    dense_input, 1, kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg), use_bias=self.use_bias)

                logits = fc + tf.reduce_sum(sparse_input, axis=-1, keep_dims=True)

            else:
                raise ValueError('dense_value_list和sparse_embedding_list全都为空，不能进行计算')
            return logits


class DNN():
    # def __init__(self, dense_feature_columns, embedding_list, units=[256, 256], use_bn=False, training=True, dropout_rate=0.5, toonedim=True):
    def __init__(self, units=[256, 256], activation='relu', l2_reg=0.0, use_bn=False, training=True, dropout_rate=0.5, toonedim=True):
        self.units = units
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        self.training = training
        self.dropout_rate = dropout_rate
        self.toonedim = toonedim
        if activation == 'relu':
            self.activation = tf.nn.relu
        elif activation == 'tanh':
            self.activation = tf.nn.tanh

    def __call__(self, dense_value_list, sparse_embedding_list):
            if len(dense_value_list) + len(sparse_embedding_list) == 0:
                raise ValueError('dense_value_list和sparse_embedding_list全都为空，不能进行计算')

            for i in range(len(dense_value_list)):
                if len(dense_value_list[i].shape) == 1:
                    dense_value_list[i] = tf.expand_dims(dense_value_list[i], axis=-1)

            dnn_input = tf.concat(dense_value_list + sparse_embedding_list, axis=-1)

            hidden = dnn_input
            for idx, u in enumerate(self.units):
                with tf.variable_scope('dnn_layer_%d' % idx):
                    hidden = tf.layers.dense(hidden, u, activation=self.activation,  kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg), kernel_initializer=tf.initializers.random_uniform)
                    if self.use_bn:
                        hidden = tf.layers.batch_normalization(hidden, training=self.training)

                    hidden = tf.layers.dropout(hidden, rate=self.dropout_rate, training=self.training)

            if self.toonedim:
                with tf.variable_scope('dnn_toonedim'):
                    logits = tf.layers.dense(hidden, 1, kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))
            else:
                logits = hidden

            return logits
        
