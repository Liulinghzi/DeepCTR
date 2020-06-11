'''
@Author: your name
@Date: 2020-05-28 16:48:36
@LastEditTime: 2020-06-04 17:15:48
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /estimator/models/neufoundr_wrapper.py
'''
import argparse
import os
import json
import sys
import pandas as pd
from data import csv2tfrecord
sys.path.append(os.path.abspath(os.path.curdir))
# # from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

parser = argparse.ArgumentParser()
# ================== 通用参数 ==================

parser.add_argument("--data_dir", type=str, default="../data/")
parser.add_argument("--input_algor", type=str, default="./output/")
parser.add_argument("--summary_save_dir", type=str, default='./log/summary/')
parser.add_argument("--model", type=str, default=None, help='选择使用的模型')

# parser.add_argument("--tfrecord_dir", type=str, default='/tmp/', help='存储tfrecord文件的地址')

# 数据格式参数
parser.add_argument("--sparse_cols", type=str, default=None)
parser.add_argument("--seq_cols", type=str, default=None)
parser.add_argument("--target", type=str, default="virginica")
parser.add_argument("--exclude", type=str, default=None)
parser.add_argument("--field_list", type=str, default=None, help='field之间以/分割，特征之间以,分割')
parser.add_argument("--remake_tfrecord", action='store_true', default=False, help='是否重新生成tfrecord')

# 训练参数
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument('--use_bn', action='store_true', default=False)

# ================== 模型特异参数 ==================
parser.add_argument('--use_deep', action='store_true', default=False) # fm & deepfm
parser.add_argument('--num_lr', type=int, default=None) # mlr
parser.add_argument('--num_crosses', type=int, default=None) # dcn
parser.add_argument('--units', type=str, default=None) # xdeepfm

args = parser.parse_args()


with open(os.path.join(args.output_path, 'config.json'), 'r') as f:
    config_json = json.load(f)


data = pd.read_csv(os.path.join(args.data_dir, 'test.csv'), engine='python')
columns = list(data.columns)

feature_spec = config_json['feature_spec']
if config_json['sparse_cols'] is None:
    sparse_cols = []
    vocab_list = []
else:
    sparse_cols = config_json['sparse_cols'].split(',')
    vocab_list = [int(v) for v in config_json['vocab_list'].split(',')]
for idx, f in enumerate(sparse_cols):
    data[f] = data[f].clip(None, vocab_list[idx])

    
    
if args.remake_tfrecord:
    csv2tfrecord(data, args.output_path, feature_spec, split=False, mode='test')


command = 'python3 %s/estimator/controller.py ' % args.model

# ================== 通用参数 ==================
command += '--tfrecord_dir %s ' % args.output_path
command += '--ckpt_save_dir %s ' %  args.input_algor
command += '--summary_save_dir %s ' %  args.summary_save_dir
command += '--mode %s ' % 'test'
command += '--project_dir %s ' %  os.path.abspath(os.path.curdir)
# 数据格式参数
if config_json['sparse_cols'] is not None:command += '--sparse_cols %s ' % config_json['sparse_cols'] 
if config_json['dense_cols'] is not None:command += '--dense_cols %s ' % config_json['dense_cols'] 
if config_json['seq_cols'] is not None: command += '--seq_cols %s ' % config_json['seq_cols']
if config_json['field_list'] is not None: command += '--field_list %s ' % config_json['field_list']
command += '--target %s ' % config_json['target']
if config_json['vocab_list'] is not None:command += '--vocab_list %s ' % config_json['vocab_list']


# 训练参数
command += '--batch_size %s ' % args.batch_size
command += '--exclude %s ' % args.exclude

# ================== 模型特异参数 ==================
if args.use_deep: command += '--use_deep '
if args.num_lr is not None: command += '--num_lr %s ' % args.num_lr
if args.num_crosses is not None: command += '--num_crosses %s ' % args.num_crosses
if args.units is not None: command += '--units %s ' % args.units
print('===========================')
print(command)
print('===========================')
os.system(command)
