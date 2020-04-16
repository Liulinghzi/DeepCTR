'''
@Author: your name
@Date: 2020-04-09 18:11:17
@LastEditTime: 2020-04-14 15:30:33
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /DeepCTR/deepctr/models/wdl.py
'''
# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Cheng H T, Koc L, Harmsen J, et al. Wide & deep learning for recommender systems[C]//Proceedings of the 1st Workshop on Deep Learning for Recommender Systems. ACM, 2016: 7-10.(https://arxiv.org/pdf/1606.07792.pdf)
"""

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense

from ..inputs import build_input_features, get_linear_logit, input_from_feature_columns, combined_dnn_input
from ..layers.core import PredictionLayer, DNN
from ..layers.utils import add_func


def WDL(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(128, 128), l2_reg_linear=1e-5,
        l2_reg_embedding=1e-5, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu',
        task='binary'):
    """Instantiates the Wide&Deep Learning architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to wide part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    features = build_input_features(
        linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, init_std, seed)


    # wide&deep两大主体部分，可以看出deepFM其实就是在里面加了一个FM

    # ======================= Wide =======================
    # deep的效果通常比wide好，wide的存在是为了解决deep的一些问题
    #   1.如果特征之间的相关矩阵特别稀疏，出现大量没有共同出现过的xi和xj，那么deep模型也能学习出embedding矩阵进行计算，但是这个embeddng矩阵是过度泛化的，
    #       含有的信息很少，实际参考意义并不大，这个时候需要wide来进行一些限制
    # TODO:交叉特征是如何实现的
    linear_logit = get_linear_logit(features, linear_feature_columns, init_std=init_std, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)

    # ======================= Deep =======================
    # deep 部分暂时能够理解，通过深度的网络对特征进行高阶的交叉，提升模型的泛化性能
    #   1.准确的来说可以解释为FM相对于LR的好处
    #       对每个特征做出一个vi向量，即使i和j没有在样本中出现过，也能够知道vivj的关系，否则如果xi和xj没有同时出现过，那么他们之间的联系是完全未知
    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                  False, seed)(dnn_input)
    dnn_logit = Dense(
        1, use_bias=False, activation=None)(dnn_out)

    final_logit = add_func([dnn_logit, linear_logit])

    # # ============== Deep FM 主体三大部分 =================

    # linear_logit = get_linear_logit(features, linear_feature_columns, init_std=init_std, seed=seed, prefix='linear',
    #                                 l2_reg=l2_reg_linear)
    # fm_logit = add_func(
    #     [
    #         FM()(concat_func(v, axis=1))
    #         for k, v in group_embedding_dict.items() if k in fm_group
    #     ]
    # )

    # dnn_input = combined_dnn_input(list(chain.from_iterable(
    #     group_embedding_dict.values())), dense_value_list)
    # dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
    #                  dnn_use_bn, seed)(dnn_input)
    # dnn_logit = tf.keras.layers.Dense(
    #     1, use_bias=False, activation=None)(dnn_output)

    # # 三部分相加
    # final_logit = add_func([linear_logit, fm_logit, dnn_logit])




    output = PredictionLayer(task)(final_logit)

    model = Model(inputs=inputs_list, outputs=output)
    return model
