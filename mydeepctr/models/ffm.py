'''
@Author: your name
@Date: 2020-05-27 10:52:57
@LastEditTime: 2020-06-09 12:50:35
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /model-building/recommend/fm/fm.py
'''
import pandas as pd
import tensorflow as tf
from inputs import SparseFeature, DenseFeature, build_input_placeholder, build_field_embedding_matrix_dict, input_from_feature_columns
from layers.baselayers import Linear, DNN

import six
import copy
import json


class FFMConfig():
    def __init__(self, dnn_feature_columns, linear_feature_columns, class_num):
        self.dnn_feature_columns = dnn_feature_columns
        self.linear_feature_columns = linear_feature_columns
        self.class_num = class_num
    
    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = FFMConfig(vocab_size=None)
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

class FFM():
    # 当前理解(2020年05月29日 星期五)：
    # FFM的基础还是同一个特征，不同embedding，公式推导中，即xixj<vfj,i  ·  vfi,j>，把特征xi和特征xj映射为了vi,vj，在不同的特征向量之间做交互
    # 首先确定特征域的数量， embedding矩阵的数量是  num_features * num_fields
    # 结构设计，双层字典，第一层的key为num_features, 第二层的key为num_fields, value为特征i，在第j个域中的embedding

    def __init__(self, model_config, inputs, labels, scope='FFM', mode=tf.estimator.ModeKeys.TRAIN):
        self.config = copy.deepcopy(model_config)
        self.inputs = inputs
        self.labels = labels

        with tf.variable_scope(scope, default_name='embeddings'):
            _, ffm_sparse_embedding_dict = input_from_feature_columns(self.inputs, self.config.linear_feature_columns, target='dnn', field=True)
            linear_dense_value_list, linear_sparse_embedding_list = input_from_feature_columns(self.inputs, self.config.linear_feature_columns, target='linear', field=False)
        
        linear_logits = Linear()(linear_dense_value_list, linear_sparse_embedding_list)
        self.logits = linear_logits

        sparse_feature_columns = [fc for fc in self.config.linear_feature_columns if isinstance(fc, SparseFeature)]
        if len(sparse_feature_columns) > 1:
            ffm_logits = self.ffm(sparse_feature_columns, ffm_sparse_embedding_dict)
            self.logits += ffm_logits
        
        if mode == tf.estimator.ModeKeys.PREDICT:
            self.loss = None
        else:
            if len(labels.shape) == 1:
                labels = tf.expand_dims(labels, axis=-1)
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=labels))

            regularity_loss = tf.losses.get_regularization_loss()
            # tf.losses.get_regularization_losses 获取的应该是list， tf.losses.get_regularization_loss是加过的
            self.loss = self.loss + regularity_loss

    def ffm(self, sparse_feature_columns, ffm_sparse_embedding_dict):
        with tf.variable_scope('ffm'):
            dot_value_list = []
            for i, f_i in enumerate(sparse_feature_columns[:-1]):
                for f_j in sparse_feature_columns[i+1:]:
                    embedding_i4fieldj = tf.nn.embedding_lookup(ffm_sparse_embedding_dict[f_i.feature_name][f_j.field_id], self.inputs[f_i.feature_name])
                    embedding_j4fieldi = tf.nn.embedding_lookup(ffm_sparse_embedding_dict[f_j.feature_name][f_i.field_id], self.inputs[f_j.feature_name])

                    # matmul乘的是张量的最后两个维度，[bs, dim]会把bs算进去变成[bs, bs]
                    # tensordot(a,b,axes=[-1,-2]和matmul等效
                    # 上述两种方式都不能实现batch的vector点乘
                    # 可实现方案
                    # 1. 对位相乘multiply成[bs, dim], reduce_sum成[bs,]
                    # 2. 扩展维度为[bs, 1, dim], [bs, dim, 1]， 然后matmul乘[bs, 1, 1], reshape为[bs,]
                    
                    dot_value = tf.multiply(embedding_i4fieldj, embedding_j4fieldi)
                    dot_value = tf.reduce_sum(dot_value, axis=-1)
                    dot_value_list.append(dot_value)
                    
            ffm_logits = sum(dot_value_list)
            ffm_logits = tf.expand_dims(ffm_logits, -1) 
            return ffm_logits   


    def get_logits(self):
        return self.logits

    def get_loss(self):
        return self.loss


feed_dict = {}


# parser = argparse.ArgumentParser()
# parser.add_argument("--data_dir", type=str, default="../data/")
# parser.add_argument("--output_path", type=str, default="./output/")
# parser.add_argument("--sparse_features", type=str, default="virginica")
# parser.add_argument("--embedding_dim", type=int, default=4)
# parser.add_argument("--target", type=str, default="virginica")
# parser.add_argument("--exclude", type=str, default="#")
# parser.add_argument("--n_estimators", type=int, default=100)
# args = parser.parse_args()


# data = pd.read_csv(args.data_dir)

# sparse_features = [feat.strip() for feat in args.sparse_features.split(',')]
# dense_features = [feat for feat in data.columns if feat not in sparse_features and feat not != args.target]
# target = args.target

# sparse_feature_columns = [SparseFeature(feature_name=feat, embedding_dim=args.embedding_dim) for feat in sparse_features]
# dense_feature_columns = [DenseFeature(feature_name=feat) for feat in dense_features]
# feature_columns = sparse_feature_columns + dense_feature_columns
# input_placeholders = build_input_placeholder(feature_columns)

# feed_dict = {feature:data[feature] for feature in input_placeholders}
# # 为什么要把placeholders放到类的外面来声明：因为需要用类名构建feed_dict
# #   其实为了保持类的封闭性，是可以把placeholder的构建放到类的里面的，并暴露一个get_placeholders的接口，返回类中构建的placeholder，构建feed_dict

# fm = FM(sparse_feature_columns, dense_feature_columns, input_placeholders)
# data = pd.DataFrame()
# fm.fit(feed_dict, data[target])
# fm.transform(data)
