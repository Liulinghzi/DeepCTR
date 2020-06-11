'''
@Author: your name
@Date: 2020-05-27 13:33:49
@LastEditTime: 2020-06-09 12:51:11
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /model-building/recommend/xdeepfm/inputs.py
'''

import tensorflow as tf
from collections import namedtuple
import copy
SparseFeature = namedtuple('SparseFeature', [
                           'feature_name', 'vocab_size', 'embedding_dim', 'field_id', 'num_fields'])
DenseFeature = namedtuple('DenseFeature', ['feature_name'])

SparseFeature.__new__.__defaults__ = (False, None, 4, None, None)
DenseFeature.__new__.__defaults__ = (False,)


def build_input_placeholder(feature_columns):
    input_placeholders_dict = {}
    for feature in feature_columns:
        if isinstance(feature, SparseFeature):
            h = tf.placeholder(dtype=tf.float32, shape=[
                               None, feature.embedding_dim])
        elif isinstance(feature, DenseFeature):
            h = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        else:
            raise ValueError('unknown')
        input_placeholders_dict[feature.feature_name] = h
    return input_placeholders_dict


def build_embedding_matrix_dict(sparse_feature_columns, target='dnn'):
    if target == 'linear':
        sfc = copy.deepcopy(sparse_feature_columns)
        for i, fc in enumerate(sfc):
            sfc[i] = sfc[i]._replace(embedding_dim=1)
    elif target != 'dnn':
        raise ValueError('target参数只能是linear或者dnn, 接收到%s' % target)

        
    embedding_matrix_dict = {}
    for feature in sparse_feature_columns:
        embedding_matrix_dict[feature.feature_name] = tf.get_variable(
            name=feature.feature_name+'_embedding_matrix_%s' % target,
            shape=[feature.vocab_size, feature.embedding_dim],
            initializer=tf.random_normal_initializer(),
            dtype=tf.float32
        )
    return embedding_matrix_dict


def build_field_embedding_matrix_dict(sparse_feature_columns):
    embedding_matrix_dict = {}
    for feature in sparse_feature_columns:
        if not isinstance(feature, SparseFeature):
            raise ValueError('只有sparse_feature才能建立embedding矩阵')
        embedding_matrix_dict[feature.feature_name] = {}
        for field in range(feature.num_fields):
            embedding_matrix_dict[feature.feature_name][field] = tf.get_variable(
                name=feature.feature_name+'_field%d'%field+'_embedding_matrix',
                shape=[feature.vocab_size, feature.embedding_dim],
                initializer=tf.random_normal_initializer(),
                dtype=tf.float32
            )
    return embedding_matrix_dict

# test = SparseFeature(embedding_dim=4)
# print(test.embedding_dim)


def input_from_feature_columns(inputs, feature_columns, target='dnn', field=False):
    dense_feature_columns = [fc for fc in feature_columns if isinstance(fc, DenseFeature)]
    sparse_feature_columns = [fc for fc in feature_columns if isinstance(fc, SparseFeature)]
    
    dense_value_list = [inputs[fc.feature_name] for fc in dense_feature_columns]
    if field:
        embedding_matrix_dict = build_field_embedding_matrix_dict(sparse_feature_columns)
        return dense_value_list, embedding_matrix_dict
        
    embedding_matrix_dict = build_embedding_matrix_dict(sparse_feature_columns, target)
    sparse_embedding_list = [tf.nn.embedding_lookup(embedding_matrix_dict[fc.feature_name], inputs[fc.feature_name]) for fc in sparse_feature_columns]

    return dense_value_list, sparse_embedding_list
