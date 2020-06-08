# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068. (https://arxiv.org/pdf/1706.06978.pdf)
"""

from tensorflow.python.keras.layers import Dense, Concatenate, Flatten
from tensorflow.python.keras.models import Model

from ..inputs import input_from_feature_columns, build_input_features, create_embedding_matrix, SparseFeat, VarLenSparseFeat, DenseFeat, embedding_lookup, get_dense_input, varlen_embedding_lookup, get_varlen_pooling_list, combined_dnn_input
from ..layers.core import DNN, PredictionLayer
from ..layers.sequence import AttentionSequencePoolingLayer
from ..layers.utils import concat_func, NoMask, add_func
from ..layers.interaction import FM


def MultiDINFM(dnn_feature_columns, history_feature_list, dnn_use_bn=False,
               dnn_hidden_units=(200, 80), dnn_activation='relu', att_hidden_size=(80, 40), att_activation="dice",
               att_weight_normalization=False, l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.0001, seed=1024,
               gender_task='binary', age_task='regression'):
    """Instantiates the Deep Interest Network architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param history_feature_list: list,to indicate  sequence sparse field
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param dnn_activation: Activation function to use in deep net
    :param att_hidden_size: list,list of positive integer , the layer number and units in each layer of attention net
    :param att_activation: Activation function to use in attention net
    :param att_weight_normalization: bool.Whether normalize the attention score of local activation unit.
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.

    """

    features = build_input_features(dnn_feature_columns)

    sparse_feature_columns = list(filter(lambda x: isinstance(
        x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    varlen_sparse_feature_columns = list(filter(lambda x: isinstance(
        x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

    history_feature_columns = []
    sparse_varlen_feature_columns = []
    history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        if feature_name in history_fc_names:
            history_feature_columns.append(fc)
        else:
            sparse_varlen_feature_columns.append(fc)

    inputs_list = list(features.values())

    embedding_dict = create_embedding_matrix(
        dnn_feature_columns, l2_reg_embedding, init_std, seed, prefix="")

    # 注意history_feature_list和history_fc_names的区别
    # history_feature_list指的是商品id，或者店铺id，这种能够构成用户行为序列的特征
    # history_fc_names值得是[商品id,商品id,商品id],这种意境构成用户行为序列的特征
    norm_emb_list = embedding_lookup(
        embedding_dict, features, sparse_feature_columns)

    fm_logit = add_func(
        [
            FM()(concat_func(v, axis=1))
            for k, v in norm_emb_list.items()
        ]
    )

    # history_feature_list查到的是一个商品的embedidng
    query_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns, history_feature_list,
                                      history_feature_list, to_list=True)

    # history_fc_names查到的是一串商品的embedding
    keys_emb_list = embedding_lookup(embedding_dict, features, history_feature_columns, history_fc_names,
                                     history_fc_names, to_list=True)

    dnn_input_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns,
                                          mask_feat_list=history_feature_list, to_list=True)
    dense_value_list = get_dense_input(features, dense_feature_columns)

    sequence_embed_dict = varlen_embedding_lookup(
        embedding_dict, features, sparse_varlen_feature_columns)
    sequence_embed_list = get_varlen_pooling_list(
        sequence_embed_dict, features, sparse_varlen_feature_columns, to_list=True)

    dnn_input_emb_list += sequence_embed_list

    keys_emb = concat_func(keys_emb_list, mask=True)
    deep_input_emb = concat_func(dnn_input_emb_list)
    query_emb = concat_func(query_emb_list, mask=True)
    hist = AttentionSequencePoolingLayer(att_hidden_size, att_activation,
                                         weight_normalization=att_weight_normalization, supports_masking=True)([
                                             query_emb, keys_emb])

    deep_input_emb = Concatenate()([NoMask()(deep_input_emb), hist])
    deep_input_emb = Flatten()(deep_input_emb)
    dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)
    gender_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn,
                        dnn_dropout, dnn_use_bn, seed)(dnn_input)
    age_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn,
                     dnn_dropout, dnn_use_bn, seed)(dnn_input)
    gender_final_logit = Dense(1, use_bias=False)(gender_output) + fm_logit
    age_final_logit = Dense(1, use_bias=False)(age_output) + fm_logit

    gender_output = PredictionLayer(
        gender_task, name='gender_output')(gender_final_logit)
    age_output = PredictionLayer(age_task, name='age_output')(age_final_logit)

    model = Model(inputs=inputs_list, outputs=[gender_output, age_output])
    return model