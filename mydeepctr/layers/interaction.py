'''
@Author: your name
@Date: 2020-06-09 11:07:18
@LastEditTime: 2020-06-09 12:44:57
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /model-building/recommend/rank/layers/interaction.py
'''

import tensorflow as tf


class FM():
    def __call__(self, sparse_embedding_list):
        with tf.variable_scope('fm'):
            # 由于tfrecord里面的变量全都是list，所以这里的embedding lookup的结果维度应该是[bs, 1, dim]  错误，实际上是[bs, dim]
            # 可能并不是list，
            fm_input = tf.stack(sparse_embedding_list, axis=1)
            # stack堆叠会多出一维， [bs, num_features, dim]

            square_of_sum = tf.square(tf.reduce_sum(fm_input, axis=1))
            sum_of_square = tf.reduce_sum(tf.square(fm_input), axis=1)
            fm_logits = 0.5 * \
                tf.reduce_sum(square_of_sum - sum_of_square, axis=1)
            fm_logits = tf.expand_dims(fm_logits, axis=-1)
            return fm_logits


