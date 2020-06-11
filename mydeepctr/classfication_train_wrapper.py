'''
@Author: your name
@Date: 2020-05-28 16:48:36
@LastEditTime: 2020-06-09 18:41:53
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
os.system('export PYTHONPATH=$PYTHONPATH:%s' % os.path.abspath(os.path.curdir))


# # from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

parser = argparse.ArgumentParser()
# ================== 通用参数 ==================
parser.add_argument("--data_dir", type=str, default=None,help='csv数据文件的地址')
parser.add_argument("--output_path", type=str, default=None, help='存储ckpt文件的地址')
parser.add_argument("--summary_save_dir", type=str, default=None, help='存储tensorboard文件的地址')
parser.add_argument("--model", type=str, default=None, help='选择使用的模型')
# parser.add_argument("--tfrecord_dir", type=str, default='/tmp/', help='存储tfrecord文件的地址')

# 数据格式参数
parser.add_argument("--col_names", type=str, default=None, help='指明哪些列名为sparse特征，特征名间以逗号分隔')
parser.add_argument("--sparse_cols", type=str, default=None, help='指明哪些列名为sparse特征，特征名间以逗号分隔')
parser.add_argument("--target", type=str, default=None, help='指明哪一列为预测目标')
parser.add_argument("--exclude", type=str, default=None, help='指明哪些列不参与计算，特征名间以逗号分隔')
parser.add_argument("--seq_cols", type=str, default=None, help='din和dien中需要设置这个参数，即需要统计为序列的特征列名') # 
parser.add_argument("--field_list", type=str, default=None, help='将特征分为几个组，只在组间做交互，组内不做交互。field之间以/分割，特征之间以,分割')
parser.add_argument("--remake_tfrecord", action='store_true', default=False, help='是否重新生成tfrecord')

# 训练参数
parser.add_argument('--units', type=str, default=None, help='dnn的层数信息')
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_epoches", type=int, default=100)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--dropout_rate", type=float, default=None)
parser.add_argument("--activation", type=str, default=None)
parser.add_argument('--use_bn', type=str, default=None)

# ================== 模型特异参数 ==================
parser.add_argument('--use_deep', action='store_true', default=False, help='决定使用fm或者deepfm')
parser.add_argument('--num_lr', type=int, default=None, help='mlr中，使用多少个lr进行加权')
parser.add_argument('--num_crosses', type=int, default=None, help='dcn中，进行多少次交互')
parser.add_argument('--method', type=str, default=None, help='pnn中，使用什么方式进行交互')
parser.add_argument('--cin_units', type=str, default=None, help='xdeepfm中，指明每一层交互中的权重矩阵数量，例如 [10, 10]')

args = parser.parse_args()
if args.use_bn is None:
    pass
elif args.use_bn.lower() == 'true':
    args.use_bn = True
elif args.use_bn.lower() == 'false':
    args.use_bn = False
else:
    raise ValueError('use_bn 只能设置True 或者False')

# data = pd.read_csv(os.path.join(args.data_dir, 'train.csv'),header=0, engine='python')
data = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))
print(data.head())
if args.col_names is not None:
    col_names = [f.strip() for f in args.col_names.strip().split(',')]
    data.columns = col_names
columns = list(data.columns)
if args.sparse_cols is not None:
    sparse_features = [f.strip() for f in args.sparse_cols.split(',')]
else:
    sparse_features = []
dense_features = [f for f in columns if f not in sparse_features and f != args.target]
vocab_list = []
for f in sparse_features:
    max_num = data[f].max() + 1
    vocab_list.append(str(max_num)) # 因为截断到max_num原来多出来的都成了


feature_spec = {}
for feat in sparse_features:
    feature_spec[feat] = 'int'
for feat in dense_features:
    feature_spec[feat] = 'float'
for feat in [args.target]:
    feature_spec[feat] ='float'   
    

config_json = {
    'model':args.model,

    'dense_cols':None if len(dense_features)==0 else ','.join(dense_features),
    'sparse_cols':args.sparse_cols,
    'seq_cols':args.seq_cols,
    'field_list':args.field_list,
    'target':args.target,
    
    'vocab_list':None if len(vocab_list)==0 else ','.join(vocab_list),
    'feature_spec':feature_spec,
    'units':args.units,
    'cin_units':args.cin_units,
    'use_bn':args.use_bn,
    'use_deep':args.use_deep,
    'method':args.method,
    'dropout_rate':args.dropout_rate,
    'activation':args.activation

}
print('===========================')
print(config_json)
print('===========================')

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)
with open(os.path.join(args.output_path, 'config.json'), 'w') as f:
    json.dump(config_json, f)


if args.remake_tfrecord:
    num_examples = csv2tfrecord(data, args.output_path, feature_spec, split=False, target=None)
    with open(os.path.join(args.output_path, 'num_examples.json'), 'w') as f:
        json.dump({'num_examples':num_examples}, f)
else:
    with open(os.path.join(args.output_path, 'num_examples.json'), 'r') as f:
        num_examples = json.load(f)['num_examples']


command = 'python3 %s/estimator/controller.py ' % args.model
# ================== 通用参数 ==================
command += '--tfrecord_dir %s ' % args.output_path
command += '--ckpt_save_dir %s ' %  args.output_path
command += '--summary_save_dir %s ' %  args.summary_save_dir
command += '--mode %s ' %  'train'
command += '--project_dir %s ' %  os.path.abspath(os.path.curdir)


# 数据格式参数
if args.sparse_cols is not None: command += '--sparse_cols %s ' % args.sparse_cols 
if len(dense_features) > 0: command += '--dense_cols %s ' % ','.join(dense_features)
if args.seq_cols is not None: command += '--seq_cols %s ' % args.seq_cols
if args.field_list is not None: command += '--field_list %s ' % args.field_list
if args.exclude is not None: command += '--exclude %s ' % args.exclude
command += '--target %s ' % args.target

if args.sparse_cols is not None: command += '--vocab_list %s ' % ','.join(vocab_list)

# 训练参数
if args.units is not None: command += '--units %s ' % args.units
if args.dropout_rate is not None: command += '--dropout_rate %s ' % args.dropout_rate
if args.activation is not None: command += '--activation %s ' % args.activation
command += '--batch_size %s ' % args.batch_size
command += '--learning_rate %s ' % args.learning_rate
command += '--num_epoches %s ' % args.num_epoches
command += '--num_examples %s ' % num_examples
if args.use_bn: command += '--use_bn '


# ================== 模型特异参数 ==================
if args.use_deep: command += '--use_deep '
if args.num_lr is not None: command += '--num_lr %s ' % args.num_lr
if args.num_crosses is not None: command += '--num_crosses %s ' % args.num_crosses
if args.cin_units is not None: command += '--cin_units %s ' % args.cin_units
if args.method is not None: command += '--method %s ' % args.method


print('===========================')
print(command)
print('===========================')
os.system(command)

