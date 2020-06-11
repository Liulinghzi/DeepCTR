'''
@Author: your name
@Date: 2020-06-11 09:29:46
@LastEditTime: 2020-06-11 14:24:12
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /DeepCTR/dataprocess/tfrecord.py
'''


import multiprocessing
# import tensorflow as tf 
# import numpy as np 
# import time 
import os
import pandas as pd
import tensorflow as tf
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder
tf.data.experimental.make_csv_dataset

def dataframe2tfrecord(output_file, examples, feature_spec):
    columns = feature_spec.keys()
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


def csv_to_tfrecord(csv_filepath, output_filedir, dense_feature_names, sparse_feature_names, label=None, split=False, mode='train',chunksize=100000):
    feature_spec = OrderedDict()
    for f in dense_feature_names:
        feature_spec[f] = 'float'
    for f in sparse_feature_names:
        feature_spec[f] = 'int'
    if label is not None:
        feature_spec[label] = 'float'
    
    chunks = pd.read_csv(csv_filepath, chunksize=chunksize)
    example_num = 0
    for idx, ck in enumerate(chunks):
        example_num += len(ck)
        if not os.path.exists(output_filedir):
            os.makedirs(output_filedir)
        dataframe2tfrecord(os.path.join(output_filedir, 'chunk_%d_%d.tfrecord' % (idx*chunksize, (idx+1)*chunksize)), ck[feature_spec.keys()].values, feature_spec)
        
    return example_num


def tfrecord_to_fn(input_files, dense_feature_names, sparse_feature_names, label=None, batch_size=128, num_epochs=100, drop_remainder=True, mode=tf.estimator.ModeKeys.TRAIN):
    if (mode == tf.estimator.ModeKeys.TRAIN or mode==tf.estimator.ModeKeys.EVAL) and label is None:
        raise ValueError('train和evaluation阶段，label不能为None')

    
    if os.path.isdir(input_files):
        input_files = [os.path.join(input_files, f) for f in  os.listdir(input_files)]
    
    name_to_features = {}
    for f in sparse_feature_names:
        name_to_features[f] = tf.io.FixedLenFeature([], tf.int64)
    for f in dense_feature_names:
        name_to_features[f] = tf.io.FixedLenFeature([], tf.float32)
    if label is not None:
        name_to_features[label] = tf.io.FixedLenFeature([], tf.float32)


    def _decode_record(record, name_to_features, label, mode):
            """Decodes a record to a TensorFlow example."""
            example = tf.parse_single_example(record, name_to_features)
            # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
            # So cast all int64 to int32.
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.to_int32(t)
                example[name] = t
            if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
                split_label = example.pop(label)
                return example, split_label
                # 如果需要把feature和label分开，需要从这里入手
    
            elif mode == tf.estimator.ModeKeys.PREDICT:
                if label is not None:
                    example.pop(label)
                return example

            else:
                raise ValueError('未知mode %s' % mode)
            

    def input_fn():
        # 闭包中需要使用变量才会把变量保存在子函数作用域
        # 闭包中不能修改父函数的不可变值

        dataset = tf.data.TFRecordDataset(input_files)
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.repeat(num_epochs)
            dataset = dataset.shuffle(buffer_size=10000)

        dataset = dataset.map(lambda record: _decode_record(record, name_to_features, label, mode), num_parallel_calls=max(4, multiprocessing.cpu_count()))
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=drop_remainder)

        return dataset

    return input_fn


def get_dataset_fromcsv(file_pattern, batch_size, num_epochs=1, column_names=None, label_name=None, select_columns=None, field_delim=',', na_value='?', num_parallel_reads=1):
    # 这里也可以用file_pattern读入一系列的csv
    # 从csv直接构建csv比较方便，但是有对应的缺点
    #   1. 无法记录seq
    #   2. 速度比tfrecord慢
    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        num_epochs=num_epochs,
        column_names=column_names,
        label_name=label_name,
        select_columns=select_columns,
        na_value=na_value,
        field_delim=field_delim,
        num_parallel_reads=num_parallel_reads
    )

    return dataset
