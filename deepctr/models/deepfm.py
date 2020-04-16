'''
@Author: your name
@Date: 2020-04-09 18:11:17
@LastEditTime: 2020-04-15 13:10:19
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /DeepCTR/deepctr/models/deepfm.py
'''
# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)

"""




from itertools import chain
import tensorflow as tf
from ..inputs import input_from_feature_columns, get_linear_logit, build_input_features, combined_dnn_input, DEFAULT_GROUP_NAME
from ..layers.core import PredictionLayer, DNN
from ..layers.interaction import FM
from ..layers.utils import concat_func, add_func
def DeepFM(linear_feature_columns, dnn_feature_columns, fm_group=[DEFAULT_GROUP_NAME], dnn_hidden_units=(128, 128),
           l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0,
           dnn_activation='relu', dnn_use_bn=False, task='binary'):
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

    features = build_input_features(
        linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())

    group_embedding_dict, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,
                                                                        init_std, seed, support_group=True)

    # ！！！重点：
    # dnn和fm的embedding矩阵是共享的group_embedding_dict
    # linear是自己的
    # ============== Deep FM 主体三大部分 =================

    # ============== 第一部分 =================
    # ============== 线性部分 =================
    # 线性部分和FM部分其实是理论FM拆分之后的两个部分，
    # 理论FM w0 + w1x1 + ... + v1v2x1x2
    # 代码线性部分就是w0 + w1x1 + ...部分，
    # 代码FM部分就是v1v2x1x2 ...部分
    linear_logit = get_linear_logit(features, linear_feature_columns, init_std=init_std, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)
    # ============== 第二部分 =================
    # ============== FM部分 =================
    # 如果看不懂这个函数，就在进行一遍公式推导
    # add_func对应最外层，针对向量每个维度的累加
    # fm_group用来对特征进行分组，只有组内的特征才会进行FM交叉，在简单情况下只有一个fm分组，暂时不管
    # 把所有特征的embedding concat起来，形成一个系数矩阵，作为FM的输入
    fm_logit = add_func(
        [
            # 这里的concat_func并不是拼接成了flatten的向量
            # 而是拼接成了
            # [
            #     [v11,v12,v13],
            #     [v21,v22,v23]
            # ]
            # 所以再算和的平方的时候，sum(axis=1)，就会计算得到
            # [v11 + v21, v12 + v22, v13 + v23]
            FM()(concat_func(v, axis=1))
            for k, v in group_embedding_dict.items() if k in fm_group
        ]
    )

    # ============== 第三部分 =================
    # ============== DNN部分 =================
    dnn_input = combined_dnn_input(list(chain.from_iterable(
        group_embedding_dict.values())), dense_value_list)
    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                     dnn_use_bn, seed)(dnn_input)
    dnn_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(dnn_output)

    # 三部分相加
    final_logit = add_func([linear_logit, fm_logit, dnn_logit])

    output = PredictionLayer(task)(final_logit)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model
