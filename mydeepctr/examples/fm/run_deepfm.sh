###
 # @Author: your name
 # @Date: 2020-05-28 18:00:27
 # @LastEditTime: 2020-06-10 12:41:26
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /estimator/run.sh
### 


python3 classfication_train_wrapper.py \
--model fm \
--data_dir /Users/liulingzhi5/dataset/movielens/ml-1m/train.csv \
--tfrecord_dir /Users/liulingzhi5/dataset/movielens/ml-1m/ \
--output_path /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/fm/estimator/log/ckpt_deepfm \
--summary_save_dir /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/fm/estimator/log/summary_deepfm \
--target ratings \
--sparse_cols users,movies,title,genres,gender,occupation \
--num_epoches 100 \
--batch_size 128 \
--learning_rate 0.001 \
--use_bn \
--use_deep \
--remake_tfrecord


python3 classfication_val_wrapper.py \
--model fm \
--data_dir /Users/liulingzhi5/dataset/movielens/ml-1m/test.csv \
--tfrecord_dir /Users/liulingzhi5/dataset/movielens/ml-1m/ \
--input_algor /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/fm/estimator/log/ckpt_deepfm \
--summary_save_dir /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/fm/estimator/log/summary_deepfm \
--batch_size 128 \
--use_deep \
--remake_tfrecord

# =================== heart ===================
python3 classfication_train_wrapper.py \
--model fm \
--data_dir /Users/liulingzhi5/dataset/heart \
--output_path /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/fm/estimator/log/deepfm_ckpt \
--summary_save_dir /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/fm/estimator/log/deepfm_summary \
--target target \
--sparse_cols sex,cp,fbs,restecg,exang,slope,ca,thal \
--num_epoches 10000 \
--batch_size 128 \
--learning_rate 0.001 \
--remake_tfrecord \
--units 4,4,4 \
--use_bn False \
--dropout_rate 0 \
--activation relu \
--use_deep



python3 classfication_val_wrapper.py \
--data_dir /Users/liulingzhi5/dataset/heart \
--input_algor /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/fm/estimator/log/deepfm_ckpt \
--summary_save_dir /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/fm/estimator/log/deepfm_summary \
--batch_size 128 \
--remake_tfrecord


python3 classfication_train_eval_wrapper.py \
--model fm \
--data_dir /Users/liulingzhi5/dataset/heart \
--output_path /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/fm/estimator/log/deepfm_ckpt \
--summary_save_dir /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/fm/estimator/log/deepfm_summary \
--target target \
--sparse_cols sex,cp,fbs,restecg,exang,slope,ca,thal \
--num_epoches 10000 \
--batch_size 128 \
--learning_rate 0.001 \
--remake_tfrecord \
--units 16,16 \
--use_bn False \
--dropout_rate 0 \
--activation relu \
--use_deep

#64,64 {'eval_accuracy': 0.3181818, 'eval_loss': 300.34933, 'eval_precision': 0.0, 'eval_recall': 0.0, 'loss': 300.34933, 'global_step': 19097}
#16,16 {'eval_accuracy': 0.3181818, 'eval_loss': 32.9699, 'eval_precision': 0.0, 'eval_recall': 0.0, 'loss': 32.9699, 'global_step': 18907}
#4,4,4 no bn {'eval_accuracy': 0.946281, 'eval_loss': 0.15555029, 'eval_precision': 0.9691358, 'eval_recall': 0.95151514, 'loss': 0.15555029, 'global_step': 23672}