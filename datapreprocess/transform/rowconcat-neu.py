'''
@Author: your name
@Date: 2020-05-25 10:31:35
@LastEditTime: 2020-05-25 10:53:10
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /dataprocess/transform/rowconcat-neu.py
'''
import pandas as pd
import argparse
import os
import shutil
import random

parser = argparse.ArgumentParser()
parser.add_argument("--up_data_dir", type=str)
parser.add_argument("--down_data_dir", type=str)
parser.add_argument("--output_data_dir", type=str)
parser.add_argument("--cols", type=str, default=None)
parser.add_argument("--target", type=str, default=None)
args = parser.parse_args()

if os.path.exists(os.path.join(args.up_data_dir,'train.csv')) and os.path.exists(os.path.join(args.down_data_dir,'train.csv')):
    mode = 'train'
    up_dataset = pd.read_csv(os.path.join(args.up_data_dir,'train.csv'))
    down_dataset = pd.read_csv(os.path.join(args.down_data_dir,'train.csv'))
elif os.path.exists(os.path.join(args.up_data_dir,'val.csv')) and os.path.exists(os.path.join(args.down_data_dir,'val.csv')):
    mode = 'val'
    up_dataset = pd.read_csv(os.path.join(args.up_data_dir,'val.csv'))
    down_dataset = pd.read_csv(os.path.join(args.down_data_dir,'val.csv'))
else:
    raise IOError('接受的文件中，既没有train.csv也没有val.csv')  

if len(up_dataset.columns) != len(down_dataset.columns):
    raise ValueError('两个输入文件列结构不同，不能进行行的拼接')
elif (up_dataset.columns == down_dataset.columns).sum() != len(up_dataset.columns):
    raise ValueError('两个输入文件列结构不同，不能进行行的拼接')

up_data = pd.read_csv(up_dataset)
down_data = pd.read_csv(down_dataset)
print('======================= 处理前 =======================')
print(up_data.head().append(up_data.tail()))
print(down_data.head().append(down_data.tail()))

out_data = up_data.append(down_data)
print('======================= 处理后 =======================')
print(out_data.head().append(out_data.tail()))

if not os.path.exists(args.output_data_dir):
  os.makedirs(args.output_data_dir)

out_data.to_csv(os.path.join(args.output_data_dir, '%s.csv' % mode), index=False)


print("行拼接完成，采样结果存储在 %s" % (args.output_data_dir))
