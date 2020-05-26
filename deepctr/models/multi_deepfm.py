# -*- coding:utf-8 -*-
'''
@Author: your name
@Date: 2020-05-07 13:30:07
@LastEditTime: 2020-05-08 17:12:52
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /code learn/DeepCTR/deepctr/models/multi_deepfm.py
'''

from itertools import chain
import tensorflow as tf
from ..inputs import input_from_feature_columns, get_linear_logit, build_input_features, combined_dnn_input, DEFAULT_GROUP_NAME
from ..layers.core import PredictionLayer, DNN
from ..layers.interaction import FM
from ..layers.utils import concat_func, add_func
def MultiDeepFM(linear_feature_columns, dnn_feature_columns, fm_group=[DEFAULT_GROUP_NAME], dnn_hidden_units=(128, 128),
           l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0,
           dnn_activation='relu', dnn_use_bn=False, gender_task='binary', age_task='multiclass'):
    """Instantiates the DeepFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param fm_group: list, group_name of features that will be used to do feature interactions.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """
    # ！！！
    # 在外部定义了一遍Input
        # 外部的Input可能只是想要通过dict.key的方式去重，以及扩展varlen
    # 在内部又定义了一遍Input
        # 这里才是真正用到的
    features = build_input_features(
        linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())

    group_embedding_dict, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,
                                                                        init_std, seed, support_group=True)

    

    linear_logit = get_linear_logit(features, linear_feature_columns, init_std=init_std, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)
    
    fm_logit = add_func(
        [
            FM()(concat_func(v, axis=1))
            for k, v in group_embedding_dict.items() if k in fm_group
        ]
    )

    dnn_input = combined_dnn_input(list(chain.from_iterable(
        group_embedding_dict.values())), dense_value_list)
    gender_dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                     dnn_use_bn, seed)(dnn_input)
    gender_dnn_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(gender_dnn_output)
        
    age_dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                     dnn_use_bn, seed)(dnn_input)
    age_dnn_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(age_dnn_output)


    # 三部分相加
    gender_final_logit = add_func([linear_logit, fm_logit, gender_dnn_logit])
    age_final_logit = add_func([linear_logit, fm_logit, age_dnn_logit])

    gender_output = PredictionLayer(gender_task, name='gender_output')(gender_final_logit)
    age_output = PredictionLayer(age_task, name='age_output')(age_final_logit)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=[gender_output, age_output])


    
    return model
