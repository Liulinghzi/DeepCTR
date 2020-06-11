'''
@Author: your name
@Date: 2020-05-27 10:52:57
@LastEditTime: 2020-06-09 18:56:30
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /model-building/recommend/PNN/PNN.py
'''
import pandas as pd
import tensorflow as tf
from inputs import SparseFeature, DenseFeature, build_input_placeholder, build_embedding_matrix_dict,input_from_feature_columns
from layers.baselayers import DNN
import six
import copy


class PNNConfig():
    def __init__(self, dnn_feature_columns, linear_feature_columns, class_num, method, use_bn, units, activation, dropout_rate):
        self.dnn_feature_columns = dnn_feature_columns
        self.linear_feature_columns = linear_feature_columns
        self.class_num = class_num
        self.method = method
        self.use_bn = use_bn
        self.units = units
        self.activation = activation
        self.dropout_rate = dropout_rate

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = PNNConfig(vocab_size=None)
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


class PNN():
    # 当前理解(2020年06月01日 星期一)：
    # 错误认识：Cross部分和FM是等效的，PNN只是单纯的把FM换成了Cross，但是FM只能生成一位，Cross可以生成任意dim位，Cross可以和DNN的结果concat
    # xdeepfm的交叉方式和PNN基本完全一样
    # FM多分类是不是可以沿着这方向去考虑
    #
    def __init__(self, model_config, inputs, labels, scope='PNN', mode=tf.estimator.ModeKeys.TRAIN):
        self.config = copy.deepcopy(model_config)
        self.inputs = inputs
        self.labels = labels
        self.mode = mode
        self.use_bn = model_config.use_bn
        self.method = model_config.method

        with tf.variable_scope(scope, default_name='embeddings'):
            dnn_dense_value_list, dnn_sparse_embedding_list = input_from_feature_columns(
                self.inputs, self.config.dnn_feature_columns, target='dnn')
            for i in range(len(dnn_dense_value_list)):
                if len(dnn_dense_value_list[i].shape) == 1:
                    dnn_dense_value_list[i] = tf.expand_dims(dnn_dense_value_list[i], axis=-1)

        
        linear_signal = tf.concat(dnn_dense_value_list+dnn_sparse_embedding_list, axis=-1)
        dnn_input = linear_signal
        if len(dnn_sparse_embedding_list) > 0:
            inner_product_signal = self.inner_product(dnn_sparse_embedding_list)
            dnn_input = tf.concat([dnn_input, linear_signal], axis=-1)


        dnn_out = DNN(units=self.config.units, activation=self.config.activation, dropout_rate=self.config.dropout_rate, use_bn=self.config.use_bn, training=mode==tf.estimator.ModeKeys.TRAIN, toonedim=False)([], [dnn_input])
        # hidden = dnn_input
        # for u in self.config.units:
        #     hidden =tf.layers.dense(hidden, u, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
        #     if self.config.use_bn:
        #         hidden = tf.layers.batch_normalization(hidden, training=mode==tf.estimator.ModeKeys.TRAIN)
        #     hidden = tf.layers.dropout(hidden, rate=0.5, training=mode==tf.estimator.ModeKeys.TRAIN)

        self.logits = tf.layers.dense(dnn_out, 1, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
        self.logits = tf.reduce_sum(self.logits, axis=-1)

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            self.loss = None
        else:
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits, labels=labels) + tf.losses.get_regularization_loss())


    def inner_product(self, embedding_list):
            num_features = len(embedding_list)
            row = []
            col = []
            for i in range(num_features-1):
                for j in range(i+1, num_features):
                    row.append(i)
                    col.append(j)
            row_embedding_list = [embedding_list[i] for i in row]
            col_embedding_list = [embedding_list[i] for i in col]
            row_concat_embedding = tf.stack(row_embedding_list, axis=1)
            col_concat_embedding = tf.stack(col_embedding_list, axis=1)

            inner_product_result = row_concat_embedding * col_concat_embedding

            inner_product_result = tf.reduce_sum(inner_product_result, axis=2)
            # 结果是batchsize， n(n-1)/2
            return inner_product_result

    def get_logits(self):
        return self.logits

    def get_loss(self):
        return self.loss


