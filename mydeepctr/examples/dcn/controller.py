'''
@Author: your name
@Date: 2020-05-27 14:53:55
@LastEditTime: 2020-06-09 18:30:57
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /model-building/recommend/estimator/dcn.py
'''
import tensorflow as tf
import os
import sys
import json
import argparse

tf.logging.set_verbosity(tf.logging.INFO)
# =================================  预先写好tfrecord =================================

parser = argparse.ArgumentParser()
parser.add_argument("--tfrecord_dir", type=str, default="../data/")
parser.add_argument("--project_dir", type=str, default='train')
parser.add_argument("--output_dir", type=str, default='train')
parser.add_argument("--mode", type=str, default='train')

parser.add_argument("--dense_cols", type=str, default=None)
parser.add_argument("--sparse_cols", type=str, default=None)
parser.add_argument("--target", type=str, default="")
parser.add_argument("--vocab_list", type=str, default=None)
parser.add_argument("--exclude", type=str, default="")

parser.add_argument("--units", type=str, default='256,256')
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--num_epoches", type=int, default=10)
parser.add_argument("--num_examples", type=int, default=100)
parser.add_argument('--use_bn', action='store_true', default=False)
parser.add_argument('--num_crosses', type=int, default=2)
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--dropout_rate', type=float, default=0)

parser.add_argument("--log_step_count_steps", type=int, default=1000)
parser.add_argument("--save_checkpoints_steps", type=int, default=1000)
parser.add_argument("--max_steps_without_decrease", type=int, default=1000)
parser.add_argument("--summary_save_dir", type=str, default='./log/summary/')
parser.add_argument("--summary_every_n_step", type=int, default=1000)
parser.add_argument("--ckpt_save_dir", type=str, default='./log/summary/')

args = parser.parse_args()
sys.path.append(args.project_dir)

from models.dcn import DCNConfig
from model import model_fn_builder
from inputs import DenseFeature, SparseFeature
from data import tfrecord2fn,csv2tfrecord

# =================================  环境配置 =================================
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
num_train_steps = args.num_examples / args.batch_size * args.num_epoches


# =================================  模型定义 =================================
if args.dense_cols is None:
    dense_features = []
else:
    dense_features = [f.strip() for f in args.dense_cols.split(',')]
if args.sparse_cols is None:
    sparse_features = []
else:
    sparse_features = [f.strip() for f in args.sparse_cols.split(',')]
if args.vocab_list is None:
    vocab_list = []
    vocab_dict = {}
else:
    vocab_list = [int(v.strip()) for v in args.vocab_list.split(',')]
    vocab_dict = {feat:vocab_list[idx] for idx, feat in enumerate(sparse_features)}

sparse_feature_columns = [SparseFeature(feature_name=feat, vocab_size=vocab_dict[feat], embedding_dim=3) for feat in sparse_features]
dense_feature_columns = [DenseFeature(feature_name=feat) for feat in dense_features]

dnn_feature_columns = dense_feature_columns + sparse_feature_columns
linear_feature_columns = dense_feature_columns + sparse_feature_columns
args.units = [int(u) for u in args.units.strip().split(',')]
model_config = DCNConfig(dnn_feature_columns, linear_feature_columns, class_num=2, num_crosses=args.num_crosses, use_bn=args.use_bn, units=args.units, dropout_rate=args.dropout_rate, activation=args.activation)
model_fn = model_fn_builder(
        model_config=model_config, 
        learning_rate=args.learning_rate,
        init_checkpoint=None,
        summary_save_dir=args.summary_save_dir, 
        summary_every_n_step=args.summary_every_n_step,
        task='binary_classification'    
)


# =================================  estimator配置 =================================
session_config = tf.ConfigProto(allow_soft_placement=True)
run_config = tf.estimator.RunConfig(
    log_step_count_steps=args.log_step_count_steps,
    save_checkpoints_steps=args.save_checkpoints_steps,
    session_config=session_config,
    model_dir=args.ckpt_save_dir
)
estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=args.ckpt_save_dir,
    params={},
    config=run_config
)

# =================================  estimator执行 =================================
# ========================  构建输入 ========================
# 配置tfrecord的数据结构格式

name2features = {}
for f in sparse_features:
    name2features[f] = tf.io.FixedLenFeature([], tf.int64)
for f in dense_features:
    name2features[f] = tf.io.FixedLenFeature([], tf.float32)
for f in [args.target]:
    name2features[f] = tf.io.FixedLenFeature([], tf.float32)
													
if args.mode == 'train':
    train_input_fn = tfrecord2fn(os.path.join(args.tfrecord_dir, 'train.tfrecord'), name2features, args.batch_size, args.num_epoches,drop_remainder=True, mode=tf.estimator.ModeKeys.TRAIN, target=args.target)
elif args.mode == 'eval':
    eval_input_fn = tfrecord2fn(os.path.join(args.tfrecord_dir, 'eval.tfrecord'), name2features, args.batch_size, args.num_epoches, drop_remainder=True, mode=tf.estimator.ModeKeys.EVAL, target=args.target)
elif args.mode == 'train_eval':
    train_input_fn = tfrecord2fn(os.path.join(args.tfrecord_dir, 'train.tfrecord'), name2features, args.batch_size, args.num_epoches,drop_remainder=True, mode=tf.estimator.ModeKeys.TRAIN, target=args.target)
    eval_input_fn = tfrecord2fn(os.path.join(args.tfrecord_dir, 'eval.tfrecord'), name2features, args.batch_size, args.num_epoches, drop_remainder=True, mode=tf.estimator.ModeKeys.EVAL, target=args.target)
elif args.mode == 'test':
    eval_input_fn = tfrecord2fn(os.path.join(args.tfrecord_dir, 'eval.tfrecord'), name2features, args.batch_size, args.num_epoches, drop_remainder=True, mode=tf.estimator.ModeKeys.PREDICT, target=args.target)
    
    

# ========================  进行训练 ========================
try:
    early_stopping_hook = tf.estimator.experimental.stop_if_no_decrease_hook(
        estimator=estimator,
        metric_name='eval_loss',
        max_steps_without_decrease=1000,
        run_every_secs=None,
        run_every_steps=1000
    )
except:
    early_stopping_hook = tf.contrib.estimator.stop_if_no_decrease_hook(
        estimator=estimator,
        metric_name='eval_loss',
        max_steps_without_decrease=1000,
        run_every_secs=None,
        run_every_steps=1000
    )

if args.mode == 'train':
    estimator.train(train_input_fn, max_steps=num_train_steps)
elif args.mode == 'eval':
    res = estimator.evaluate(eval_input_fn)
    print(res)
    res = {k:float(res[k]) for k in res}
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, 'result.json'), 'w') as f:
        json.dump(res, f)
        
elif args.mode == 'train_eval':

    tf.estimator.train_and_evaluate(
        estimator,
        train_spec=tf.estimator.TrainSpec(train_input_fn, max_steps=num_train_steps,
                                            hooks=[early_stopping_hook]),
        eval_spec=tf.estimator.EvalSpec(eval_input_fn, steps=1000))
