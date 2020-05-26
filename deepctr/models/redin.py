'''
@Author: your name
@Date: 2020-05-22 15:11:54
@LastEditTime: 2020-05-22 15:15:20
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /DeepCTR/deepctr/models/redin.py
'''
import tensorflow as tf

def LocalAttetionUnit(query, key):
    # query [bs, 1, dim]
    # key [bs, T, dim]
    query = tf.tile(query, [1, key.shape[1], 1])
    
    att_in = tf.concat([
        query, key, query - key, query * key
    ], axis=-1)

    att_out = tf.layers.dense(att_in, 100)
    # [bs, T, 4*dim]


    pass