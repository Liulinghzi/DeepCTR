'''
@Author: your name
@Date: 2020-05-27 10:52:57
@LastEditTime: 2020-06-09 18:27:07
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /model-building/recommend/fm/fm.py
========================================================================================================
当前理解(2020年05月27日 星期三)：
FM的基础还是embedding，公式推导中，即xixj<vi·vj>，把特征xi和特征xj映射为了vi,vj，在不同的特征向量之间做交互
所有特征之间做的交互，实际上是一个方阵的下三角区域（不包含对角线， 对角线是自己和自己的交互，不再fm的计算范围之内），整个矩阵是关于对角线重复的，所以为了方便计算，算了(整个矩阵 - 额外一条对角线)/2
    整个矩阵：两个取和函数可以进行拆分，变成两个求和函数相乘，即所有vi，先取和，再平方（这里的平方其实是两个向量点积，每个元素平方后还需要加和）。
    对角线：向量自己的平方（这里的平方其实是两个向量点积，每个元素平方后还需要加和）在求和
所以结果为 tf.squrae(sum(embedding_list)) - sum(tf.square(embedding_list))
    embedding_list 为

    batchsize *[
                [v11,v12,v13],
                [v21,v22,v23]
            ]
    sum希望的结果是 batchsize * [v11+v21, v12+v22, v13+v33], 所以是reduce_sum(axis=1),把1轴加掉
希望的使用方式
tf.reduce_sum(tf.square(tf.reduce_sum(embedding_list, axis=1))) - tf.reduce_sum(tf.reduce_sum(tf.square(embedding_list), axis=1))
最外层的tf.reduce_sum可以合并
tf.reduce_sum(tf.square(tf.reduce_sum(embedding_list, axis=1)) - tf.reduce_sum(tf.square(embedding_list), axis=1))

========================================================================================================
理解更新(2020年06月08日 星期一)
deepctr中的代码架构中，linear部分和deep部分分别建立了不同的embedding矩阵
'''
import pandas as pd
import tensorflow as tf
from inputs import SparseFeature, DenseFeature, build_input_placeholder, build_embedding_matrix_dict, input_from_feature_columns
from layers.baselayers import Linear, DNN
from layers.interaction import FM as fm
import json
import six
import copy


class FMConfig():
    def __init__(self, dnn_feature_columns, linear_feature_columns, class_num, use_deep, use_bn, activation, dropout_rate, units=None):
        self.dnn_feature_columns = dnn_feature_columns
        self.linear_feature_columns = linear_feature_columns
        self.class_num = class_num
        self.use_deep = use_deep
        self.use_bn = use_bn
        self.units = units
        self.activation = activation
        self.dropout_rate = dropout_rate
        
    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = FMConfig(vocab_size=None)
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


class FM():
    def __init__(self, model_config, inputs, labels, scope='FM', mode=tf.estimator.ModeKeys.TRAIN):
        self.config = copy.deepcopy(model_config)
        self.inputs = inputs
        self.labels = labels
        self.mode = mode
        self.use_deep = model_config.use_deep
        self.use_bn = model_config.use_bn

        with tf.variable_scope(scope, default_name='embeddings'):
            dnn_dense_value_list, dnn_sparse_embedding_list = input_from_feature_columns(
                self.inputs, self.config.dnn_feature_columns, target='dnn')
            linear_dense_value_list, linear_sparse_embedding_list = input_from_feature_columns(
                self.inputs, self.config.linear_feature_columns, target='linear')

        
        linear_logits = Linear()(linear_dense_value_list,
                                 linear_sparse_embedding_list)

        self.logits = linear_logits

        if len(dnn_sparse_embedding_list) >0:
            fm_logits = fm()(dnn_sparse_embedding_list)
            self.logits += fm_logits

        if self.use_deep:
            deep_logits = DNN(units=self.config.units, activation=self.config.activation, dropout_rate=self.config.dropout_rate, use_bn=self.config.use_bn, training=tf.estimator.ModeKeys.TRAIN == mode, toonedim=True)(
                dnn_dense_value_list, dnn_sparse_embedding_list)
            self.logits += deep_logits

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            self.loss = None
        else:
            if len(labels.shape) == 1:
                labels = tf.expand_dims(labels, axis=-1)
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits, labels=labels) + tf.losses.get_regularization_loss())

    def get_logits(self):
        return self.logits

    def get_loss(self):
        return self.loss
