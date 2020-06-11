###
 # @Author: your name
 # @Date: 2020-05-28 18:00:27
 # @LastEditTime: 2020-06-09 18:46:14
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /estimator/run.sh
### 





python3 classfication_train_wrapper.py \
--model mlr \
--data_dir /Users/liulingzhi5/dataset/heart \
--output_path /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/mlr/estimator/log/ckpt_mlr \
--summary_save_dir /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/mlr/estimator/log/summary_mlr \
--target target \
--sparse_cols sex,cp,fbs,restecg,exang,slope,ca,thal \
--num_epoches 10000 \
--batch_size 128 \
--learning_rate 0.001 \
--num_lr 4 \
--remake_tfrecord


python3 classfication_val_wrapper.py \
--data_dir /Users/liulingzhi5/dataset/heart \
--input_algor /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/mlr/estimator/log/ckpt_mlr \
--summary_save_dir /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/mlr/estimator/log/summary_mlr \
--batch_size 128 \
--remake_tfrecord




# ================ movielens ================ 










python3 classfication_train_wrapper.py \
--model mlr \
--data_dir /Users/liulingzhi5/dataset/movielens/ml-1m/train.csv \
--tfrecord_dir /Users/liulingzhi5/dataset/movielens/ml-1m/ \
--output_path /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/mlr/estimator/log/ckpt_mlr \
--summary_save_dir /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/mlr/estimator/log/summary_mlr \
--target ratings \
--sparse_cols users,movies,title,genres,gender,occupation \
--num_epoches 100 \
--batch_size 128 \
--learning_rate 0.001 \
--num_lr 4 \
--remake_tfrecord


python3 classfication_val_wrapper.py \
--model mlr \
--tfrecord_dir /Users/liulingzhi5/dataset/movielens/ml-1m/ \
--data_dir /Users/liulingzhi5/dataset/movielens/ml-1m/test.csv \
--input_algor /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/mlr/estimator/log/ckpt_mlr \
--summary_save_dir /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/mlr/estimator/log/summary_mlr \
--batch_size 128 \
--num_lr 4 \
--remake_tfrecord





python3 classfication_test_wrapper.py \
--model mlr \
--tfrecord_dir /Users/liulingzhi5/dataset/movielens/ml-1m/ \
--data_dir /Users/liulingzhi5/dataset/movielens/ml-1m/test.csv \
--input_algor /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/mlr/estimator/log/ckpt_mlr \
--summary_save_dir /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/mlr/estimator/log/summary_mlr \
--batch_size 128 \
--num_lr 4 \
--remake_tfrecord


python3 classfication_train_eval_wrapper.py \
--model mlr \
--tfrecord_dir /Users/liulingzhi5/dataset/movielens/ml-1m/ \
--data_dir /Users/liulingzhi5/dataset/movielens/ml-1m/test.csv \
--input_algor /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/mlr/estimator/log/ckpt_mlr \
--summary_save_dir /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/mlr/estimator/log/summary_mlr \
--batch_size 128 \
--num_lr 4 \
--remake_tfrecord
