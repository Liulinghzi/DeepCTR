'''
@Author: your name
@Date: 2020-05-27 10:52:57
@LastEditTime: 2020-06-09 18:54:42
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /model-building/recommend/dcn/dcn.py
'''
import pandas as pd
import tensorflow as tf
from inputs import SparseFeature, DenseFeature, build_input_placeholder, build_embedding_matrix_dict,input_from_feature_columns
from layers.baselayers import DNN
import six
import copy


class DCNConfig():
    def __init__(self, dnn_feature_columns, linear_feature_columns, class_num, num_crosses, use_bn, activation, dropout_rate, units):
        self.dnn_feature_columns = dnn_feature_columns
        self.linear_feature_columns = linear_feature_columns
        self.class_num = class_num
        self.num_crosses = num_crosses
        self.use_bn = use_bn
        self.units = units
        self.activation = activation
        self.dropout_rate = dropout_rate

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = DCNConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DCN():
    # 当前理解(2020年06月01日 星期一)：
    # 错误认识：Cross部分和FM是等效的，DCN只是单纯的把FM换成了Cross，但是FM只能生成一位，Cross可以生成任意dim位，Cross可以和DNN的结果concat
    # xdeepfm的交叉方式和DCN基本完全一样
    # FM多分类是不是可以沿着这方向去考虑
    #
    def __init__(self, model_config, inputs, labels, scope='DCN', mode=tf.estimator.ModeKeys.TRAIN):
        self.config = copy.deepcopy(model_config)
        self.inputs = inputs
        self.labels = labels
        self.mode = mode
        self.use_bn = model_config.use_bn
        self.num_crosses = model_config.num_crosses

        with tf.variable_scope(scope, default_name='embeddings'):
            dnn_dense_value_list, dnn_sparse_embedding_list = input_from_feature_columns(
                self.inputs, self.config.dnn_feature_columns, target='dnn')
        
        if len(dnn_sparse_embedding_list) > 0:
            cross_logits = self.cross(dnn_sparse_embedding_list)
            deep_logits = DNN(units=self.config.units, activation=self.config.activation, dropout_rate=self.config.dropout_rate, use_bn=self.config.use_bn, training=mode==tf.estimator.ModeKeys.TRAIN, toonedim=False)(dnn_dense_value_list, dnn_sparse_embedding_list)

            logits = tf.concat([cross_logits, deep_logits], axis=-1)
        else:
            deep_logits = DNN(units=self.config.units, activation=self.config.activation, dropout_rate=self.config.dropout_rate, use_bn=self.config.use_bn, training=mode==tf.estimator.ModeKeys.TRAIN, toonedim=False)(dnn_dense_value_list, dnn_sparse_embedding_list)
            logits= deep_logits

        self.logits = tf.layers.dense(logits, 1, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
        self.logits = tf.reduce_sum(self.logits, axis=-1)

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            self.loss = None
        else:
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits, labels=labels) + tf.losses.get_regularization_loss())


    def cross(self, embedding_list):
        # 由于tfrecord里面的变量全都是list，所以这里的embedding lookup的结果维度应该是[bs, 1, dim]  错误，实际上是[bs, dim]
        # 可能并不是list，
        with tf.variable_scope('crossnet'):

            cross_input = tf.concat(embedding_list, axis=1)
            cross_input = tf.expand_dims(cross_input, axis=2)
            # [bs, dim * num_features, 1]

            xi = cross_input
            for i in range(self.config.num_crosses):
                # 这里应该用矩阵还是用向量，感觉矩阵只是白白增加的参数量
                cross_weight = tf.get_variable(name='cross_weight_%s' % (i), shape=(cross_input.shape[1], 1), dtype=tf.float32, initializer=tf.random_uniform_initializer())
                cross_bias = tf.get_variable(name='cross_bias_%s' % (i), shape=(cross_input.shape[1], 1), dtype=tf.float32, initializer=tf.random_uniform_initializer())
                
                cross = tf.matmul(cross_input, xi, transpose_b=True) # bs, dim*num_features, dim*num_features
                # cross = tf.matmul(cross, cross_weight)  # bs, dim*num_features, 1  tf1.12不支持？？ 也许是包改过没更新
                # print(cross)
                # print(cross_weight )
                # print(cross)
                # exit()
                cross = tf.tensordot(cross, cross_weight, axes=(2, 0))
                xi = xi + cross + cross_bias

            xi = tf.reduce_sum(xi, axis=-1)
            
            return xi

    def get_logits(self):
        return self.logits

    def get_loss(self):
        return self.loss


