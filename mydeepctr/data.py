'''
@Author: your name
@Date: 2020-05-27 14:54:07
@LastEditTime: 2020-06-04 18:18:11
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /model-building/recommend/estimator/convert2tfrecord.py
'''
import tensorflow as tf
import pandas as pd
import numpy as np
from collections import namedtuple, OrderedDict, Counter
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import multiprocessing
import gc
import os
import multiprocessing

def dataframe2tfrecord(output_file, examples, feature_spec, columns):
    writer = tf.python_io.TFRecordWriter(output_file)
    for idx, example in enumerate(examples):
        if idx % 10000 == 0:
            tf.logging.info('Writing example %d of %d' % (idx, len(examples)))

        features = OrderedDict()
        for num_feat, feat in enumerate(columns):
            if feature_spec[feat] == 'str':
                features[feat] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[str(example[num_feat])]))
            elif feature_spec[feat] == 'int':
                features[feat] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[int(example[num_feat])]))
            elif feature_spec[feat] == 'float':
                features[feat] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=[float(example[num_feat])]))
            else:
                raise ValueError('类型%s不可识别, 只接受 str  int  float' % feature_spec[feat])

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())

    writer.close()    


def csv2tfrecord(data, output_dir, feature_spec, split=False, target=None, mode='train'):
    df = data
    dense = [f for f in feature_spec if (feature_spec[f] == 'float' and f != target)]
    sparse = [f for f in feature_spec if (feature_spec[f] == 'int')]

    # df[dense] = MinMaxScaler().fit_transform(df[dense])
    # for f in sparse:
    #     df[f] = LabelEncoder().fit_transform(df[f])
    # df = df.sample(frac=1)

    columns = list(df.columns)
    print('====================')
    print(columns)
    print(list(feature_spec.keys()))
    print('====================')
    examples = df.values
    example_num = len(df)
    if target is not None:
        print('=====================================')
        print(Counter(df[int(example_num * 0.8):][target]))
        print('=====================================')
    del df
    gc.collect()
    if mode == 'train':
        if split:
            dataframe2tfrecord(os.path.join(output_dir,'train.tfrecord'), examples[: int(example_num * 0.8)], feature_spec, columns)
            dataframe2tfrecord(os.path.join(output_dir,'eval.tfrecord'), examples[int(example_num * 0.8):], feature_spec, columns)
            return int(example_num * 0.8)
        dataframe2tfrecord(os.path.join(output_dir, 'train.tfrecord'), examples[: int(example_num * 0.8)], feature_spec, columns)
    elif mode == 'test':
        dataframe2tfrecord(os.path.join(output_dir, 'test.tfrecord'), examples[: int(example_num * 0.8)], feature_spec, columns)
    elif mode == 'eval':
        dataframe2tfrecord(os.path.join(output_dir, 'eval.tfrecord'), examples[: int(example_num * 0.8)], feature_spec, columns)
        
    return example_num


def tfrecord2fn(input_file, name2features, batch_size, num_epochs,  drop_remainder=True, mode=tf.estimator.ModeKeys.TRAIN, target=None):
    if (mode == tf.estimator.ModeKeys.TRAIN or mode==tf.estimator.ModeKeys.EVAL) and target is None:
        raise ValueError('train和evaluation阶段，target不能为None')

    def _decode_record(record, name2features):
        example = tf.parse_single_example(record, name2features)

        for name in list(example.keys()):
            value = example[name]
            if value.dtype == tf.int64:
                value = tf.to_int32(value)
            example[name] = value
        return example

    def input_fn():
        dataset = tf.data.TFRecordDataset(input_file)
        if mode==tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.repeat()
            dataset = dataset.shuffle(buffer_size=batch_size*100)


        dataset = tf.contrib.learn.read_batch_features(
            input_file, batch_size, name2features, tf.TFRecordReader,
            num_epochs=num_epochs, reader_num_threads=min(4, multiprocessing.cpu_count()))

        # dataset的每一行是一个record， apply就是对每个record进行操作
        # dataset = dataset.map(
        #         map_func=lambda record: _decode_record(record, name2features),
        #     )
        # dataset = dataset.batch(
        #         batch_size=batch_size,
        #         drop_remainder=drop_remainder
        #     )

        label = None
        if target is not None:
            label = dataset.pop(target)

        # if mode == tf.estimator.ModeKeys.PREDICT:
        #     return dataset
        # else:
        #     return dataset, label
        return dataset, label

    return input_fn
