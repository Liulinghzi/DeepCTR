'''
@Author: your name
@Date: 2020-05-27 10:52:57
@LastEditTime: 2020-06-10 10:32:03
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /model-building/recommend/mlr/mlr.py
'''
import pandas as pd
import tensorflow as tf
from inputs import SparseFeature, DenseFeature, build_input_placeholder, build_embedding_matrix_dict,input_from_feature_columns
from layers.baselayers import Linear, DNN
import six
import copy


class MLRConfig():
    def __init__(self, dnn_feature_columns, linear_feature_columns, class_num, num_lr):
        self.dnn_feature_columns = dnn_feature_columns
        self.linear_feature_columns = linear_feature_columns
        self.class_num = class_num
        self.num_lr = num_lr

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = MLRConfig(vocab_size=None)
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


class MLR():

    def __init__(self, model_config, inputs, labels, scope='MLR', mode=tf.estimator.ModeKeys.TRAIN):
        self.config = copy.deepcopy(model_config)
        self.inputs = inputs
        self.labels = labels
        self.mode = mode
        self.num_lr = model_config.num_lr

        with tf.variable_scope(scope, default_name='embeddings'):
            linear_dense_value_list, linear_sparse_embedding_list = input_from_feature_columns(
                self.inputs, self.config.linear_feature_columns, target='linear')

        lr_logits_list = []
        for i in range(self.num_lr):
            with tf.variable_scope('lr%d'%i, default_name='linear'):
                lr_logits = Linear()(linear_dense_value_list, linear_sparse_embedding_list)
                lr_logits_list.append(lr_logits)

        logits = tf.concat(lr_logits_list, axis=-1)
        self.logits = tf.layers.dense(logits, 1, name='logits_weight')
        self.logits = tf.reduce_mean(self.logits, axis=-1)

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            self.loss = None
        else:
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits, labels=labels) + tf.losses.get_regularization_loss())


    def get_logits(self):
        return self.logits

    def get_loss(self):
        return self.loss

