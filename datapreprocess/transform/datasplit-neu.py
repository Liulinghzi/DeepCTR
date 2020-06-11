'''
@Author: your name
@Date: 2020-05-25 10:29:25
@LastEditTime: 2020-05-25 10:29:55
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /dataprocess/transform/datasplit-neu.py
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
parser.add_argument("--cols", type=str, default=None)
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


if args.cols == 'all':
  cols = [col for col in train_data.columns if col != args.target]
else:
  cols = [c.strip() for c in args.cols.split(',')]
  for col in cols:
    if col not in train_data.columns:
      raise ValueError('输入的列%s在数据集中不存在，请检查' % col)
if args.target not in [cols]:
  cols.append(args.target)

train_data = train_data[cols]
val_data = val_data[cols]
print('======================= 处理后 =======================')
print(train_data.head().append(train_data.tail()))
print(val_data.head().append(val_data.tail()))

if not os.path.exists(args.output_train_dir):
  os.makedirs(args.output_train_dir)

if not os.path.exists(args.output_val_dir):
  os.makedirs(args.output_val_dir)

train_data.to_csv(os.path.join(args.output_train_dir, 'train.csv'), index=False)
val_data.to_csv(os.path.join(args.output_val_dir, 'val.csv'), index=False)

print("去除重复行完成，采样结果存储在 %s和%s" % (args.output_train_dir, args.output_val_dir))
