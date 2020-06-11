'''
@Author: your name
@Date: 2020-05-25 10:12:10
@LastEditTime: 2020-05-25 11:03:46
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /dataprocess/transform/rename.py
'''

import pandas as pd
import argparse
import os
import shutil
import random

parser = argparse.ArgumentParser()
parser.add_argument("--train_data_dir", type=str)
parser.add_argument("--val_data_dir", type=str)
parser.add_argument("--output_train_dir", type=str)
parser.add_argument("--output_val_dir", type=str)
parser.add_argument("--old_cols", type=str)
parser.add_argument("--new_cols", type=str)
parser.add_argument("--target", type=str, default=None)
args = parser.parse_args()

train_dataset = os.path.join(args.train_data_dir,'train.csv')
if not os.path.exists(train_dataset):
  print("ERROR: train.csv is not exists!")
  exit()
  
val_dataset = os.path.join(args.val_data_dir,'val.csv')
if not os.path.exists(val_dataset):
  print("ERROR: val.csv is not exists!")
  exit()

train_data = pd.read_csv(train_dataset)
val_data = pd.read_csv(val_dataset)
print('======================= 处理前 =======================')
print(train_data.head().append(train_data.tail()))
print(val_data.head().append(val_data.tail()))


if args.old_cols == 'all':
  old_cols = [old_cols for old_cols in train_data.columns]
else:
  old_cols = [c.strip() for c in args.old_cols.split(',')]
  for col in old_cols:
    if col not in train_data.columns:
      raise ValueError('输入的列%s在数据集中不存在，请检查' % col)
new_cols = [c.strip() for c in args.new_cols.split(',')]

mapper = {old_cols[i]:new_cols[i] for i,c in enumerate(old_cols)}

train_data = train_data.rename(mapper, axis='columns')
val_data = val_data.rename(mapper, axis='columns')
print('======================= 处理后 =======================')
print(train_data.head().append(train_data.tail()))
print(val_data.head().append(val_data.tail()))

if not os.path.exists(args.output_train_dir):
  os.makedirs(args.output_train_dir)

if not os.path.exists(args.output_val_dir):
  os.makedirs(args.output_val_dir)

train_data.to_csv(os.path.join(args.output_train_dir, 'train.csv'), index=False)
val_data.to_csv(os.path.join(args.output_val_dir, 'val.csv'), index=False)

print("列重命名完成，采样结果存储在 %s和%s" % (args.output_train_dir, args.output_val_dir))
