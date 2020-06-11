'''
@Author: your name
@Date: 2020-05-25 10:31:42
@LastEditTime: 2020-05-25 11:03:34
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /dataprocess/transform/colconcat-neu.py
'''

import pandas as pd
import argparse
import os
import shutil
import random

parser = argparse.ArgumentParser()
parser.add_argument("--left_data_dir", type=str)
parser.add_argument("--right_data_dir", type=str)
parser.add_argument("--output_data_dir", type=str)
parser.add_argument("--cols", type=str, default=None)
parser.add_argument("--target", type=str, default=None)
args = parser.parse_args()

if os.path.exists(os.path.join(args.left_data_dir, 'train.csv')) and os.path.exists(os.path.join(args.right_data_dir, 'train.csv')):
    mode = 'train'
    left_dataset = pd.read_csv(os.path.join(args.left_data_dir, 'train.csv'))
    right_dataset = pd.read_csv(os.path.join(args.right_data_dir, 'train.csv'))
elif os.path.exists(os.path.join(args.left_data_dir, 'val.csv')) and os.path.exists(os.path.join(args.right_data_dir, 'val.csv')):
    mode = 'val'
    left_dataset = pd.read_csv(os.path.join(args.left_data_dir, 'val.csv'))
    right_dataset = pd.read_csv(os.path.join(args.right_data_dir, 'val.csv'))
else:
    raise IOError('接受的文件中，既没有train.csv也没有val.csv')

if len(left_dataset) != len(right_dataset):
    raise ValueError('两个输入文件行数不同，不能进行列的拼接')

left_data = pd.read_csv(left_dataset)
right_data = pd.read_csv(right_dataset)


print('======================= 处理前 =======================')
print(left_data.head().append(left_data.tail()))
print(right_data.head().append(right_data.tail()))

left_data.columns = [col+'_left' for col in left_data.columns]
right_data.columns = [col+'_right' for col in right_data.columns]
out_data = pd.concat([left_data. right_data], axis=1)
print('======================= 处理后 =======================')
print(out_data.head().append(out_data.tail()))

if not os.path.exists(args.output_data_dir):
    os.makedirs(args.output_data_dir)

out_data.to_csv(os.path.join(args.output_data_dir,
                             '%s.csv' % mode), index=False)


print("列拼接完成，采样结果存储在 %s" % (args.output_data_dir))
