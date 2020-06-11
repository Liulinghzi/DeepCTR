'''
@Author: your name
@Date: 2020-05-25 11:01:09
@LastEditTime: 2020-05-25 11:18:00
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /dataprocess/func_base_transform.py
'''

import pandas as pd
import argparse
import os
import shutil
import numpy as np
from scipy import stats, special

parser = argparse.ArgumentParser()
parser.add_argument("--train_data_dir", type=str)
parser.add_argument("--val_data_dir", type=str)
parser.add_argument("--output_train_dir", type=str)
parser.add_argument("--output_val_dir", type=str)
parser.add_argument("--cols", type=str, default=None)
parser.add_argument("--target", type=str, default=None)
parser.add_argument("--lambda_function", type=str)
args = parser.parse_args()

train_dataset = os.path.join(args.train_data_dir, 'train.csv')
if not os.path.exists(train_dataset):
    print("ERROR: train.csv is not exists!")
    exit()

val_dataset = os.path.join(args.val_data_dir, 'val.csv')
if not os.path.exists(val_dataset):
    print("ERROR: val.csv is not exists!")
    exit()

train_data = pd.read_csv(train_dataset)
val_data = pd.read_csv(val_dataset)
print('======================= 处理前 =======================')
print('训练集：', train_data.shape)
print('测试集：', val_data.shape)

len_train_data = len(train_data)
concat_data = pd.concat([train_data, val_data], axis=0).reset_index(drop=True)

if args.cols == 'all':
    cols = [col for col in concat_data.columns if concat_data[col].dtype != 'object']
else:
    cols = [c.strip() for c in args.cols.split(',')]

for col in cols:
    if col not in concat_data.columns:
        raise ValueError('输入的列%s在数据集中不存在，请检查' % col)
    if concat_data[col].isna().sum() != 0:
        raise ValueError('输入的列%s含有空值，请先使用缺失值填充或者缺失值删除进行处理' % col)

    concat_data[col] = concat_data[col].map(eval(args.lambda_function))

train_data = concat_data[:len_train_data].reset_index(drop=True)
val_data = concat_data[len_train_data:].reset_index(drop=True)
print('======================= 处理后 =======================')
print('训练集：', train_data.shape)
print('测试集：', val_data.shape)
print('======================= train =======================')
print(train_data)
print('======================= val =======================')
print(val_data)

if not os.path.exists(args.output_train_dir):
    os.makedirs(args.output_train_dir)

if not os.path.exists(args.output_val_dir):
    os.makedirs(args.output_val_dir)

train_data.to_csv(os.path.join(
    args.output_train_dir, 'train.csv'), index=False)
val_data.to_csv(os.path.join(args.output_val_dir, 'val.csv'), index=False)

print("自定义函数特征转换完成，结果存储在 %s和%s" % (args.output_train_dir, args.output_val_dir))


