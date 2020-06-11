'''
@Author: your name
@Date: 2020-04-09 18:11:17
@LastEditTime: 2020-06-11 14:23:02
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /DeepCTR/examples/run_multivalue_movielens.py
'''
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import sys
import os
sys.path.append('/Users/liulingzhi5/Desktop/code learn/DeepCTR')
from deepctr.inputs import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names
from deepctr.models import DeepFM
from datapreprocess.tfrecord import tfrecord_to_fn, csv_to_tfrecord, get_dataset_fromcsv


if __name__ == "__main__":

    dense_feature_names = ['age','trestbps','chol','thalach','oldpeak']
    sparse_feature_names = ['sex','cp','fbs','restecg','exang','slope','ca','thal']
    label = 'target'
    columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']
    train_filename = 'examples/data_sample/heart/train.csv'
    val_filename = 'examples/data_sample/heart/val.csv'
    test_filename = 'examples/data_sample/heart/val.csv'
    save_dir = 'examples/data_sample/heart/'

    vocab_dict =  {
        'sex':2,
        'cp':4,
        'fbs':2,
        'restecg':3,
        'exang':2,
        'slope':3,
        'ca':5,
        'thal':4
    }
    csv_to_tfrecord(train_filename, output_filedir=os.path.join(save_dir, 'train_tfrecord'), dense_feature_names=dense_feature_names, sparse_feature_names=sparse_feature_names, label=label)
    csv_to_tfrecord(val_filename, output_filedir=os.path.join(save_dir, 'val_tfrecord'), dense_feature_names=dense_feature_names, sparse_feature_names=sparse_feature_names, label=label)
    csv_to_tfrecord(test_filename, output_filedir=os.path.join(save_dir, 'test_tfrecord'), dense_feature_names=dense_feature_names, sparse_feature_names=sparse_feature_names, label=None)

    dense_feature_columns = [DenseFeat(feat) for feat in dense_feature_names]

    sparse_feature_columns = [SparseFeat(feat, vocab_dict[feat], embedding_dim=4)
                              for feat in sparse_feature_names]

    linear_feature_columns = dense_feature_columns + sparse_feature_columns
    dnn_feature_columns = dense_feature_columns + sparse_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    model = DeepFM(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=[4,4], task='binary')

    model.compile("adam", "binary_crossentropy", metrics=['accuracy'], )

    tensorboard_callback = tf.keras.callbacks.TensorBoard()
    train_input_fn = tfrecord_to_fn('/Users/liulingzhi5/Desktop/code learn/DeepCTR/heart_train', dense_feature_names=dense_feature_names, sparse_feature_names=sparse_feature_names, label=label, mode=tf.estimator.ModeKeys.TRAIN)
    val_input_fn = tfrecord_to_fn('/Users/liulingzhi5/Desktop/code learn/DeepCTR/heart_val', dense_feature_names=dense_feature_names, sparse_feature_names=sparse_feature_names, label=label, mode=tf.estimator.ModeKeys.EVAL)
    pred_input_fn = tfrecord_to_fn('/Users/liulingzhi5/Desktop/code learn/DeepCTR/heart_val', dense_feature_names=dense_feature_names, sparse_feature_names=sparse_feature_names, label=None, mode=tf.estimator.ModeKeys.PREDICT)
    train_datasets = train_input_fn()
    val_datasets = val_input_fn()
    pred_datasets = pred_input_fn()
        
    history = model.fit(train_datasets, epochs=10, verbose=1)
    print('train finish')
    res = model.evaluate(val_datasets)
    print(res)
    res = model.predict(pred_datasets)
    print(res)
