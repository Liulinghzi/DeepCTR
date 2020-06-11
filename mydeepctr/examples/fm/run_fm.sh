###
 # @Author: your name
 # @Date: 2020-05-28 18:00:27
 # @LastEditTime: 2020-06-10 13:28:41
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /estimator/run.sh
### 


python3 examples/fm/controller.py \
--data_dir /Users/liulingzhi5/dataset/movielens/ml-1m/ \
--output_path /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/fm/estimator/log/ckpt_fm \
--summary_save_dir /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/fm/estimator/log/summary_fm \
--target rating1 \
--sparse_cols users,movies,title,genres,gender,occupation \
--num_epoches 100 \
--batch_size 128 \
--learning_rate 0.001 \
--use_bn 1 \
--remake_tfrecord


python3 classfication_val_wrapper.py \
--model fm \
--data_dir /Users/liulingzhi5/dataset/movielens/ml-1m/test.csv \
--tfrecord_dir /Users/liulingzhi5/dataset/movielens/ml-1m/ \
--input_algor /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/fm/estimator/log/ckpt_fm \
--summary_save_dir /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/fm/estimator/log/summary_fm \
--batch_size 128 \
--remake_tfrecord

# =================== heart ===================
python3 classfication_train_wrapper.py \
--model fm \
--data_dir /Users/liulingzhi5/dataset/heart \
--output_path /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/fm/estimator/log/ckpt \
--summary_save_dir /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/fm/estimator/log/summary \
--target target \
--sparse_cols sex,cp,fbs,restecg,exang,slope,ca,thal \
--num_epoches 100 \
--batch_size 128 \
--learning_rate 0.001 \
--remake_tfrecord



python3 classfication_val_wrapper.py \
--data_dir /Users/liulingzhi5/dataset/heart \
--input_algor /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/fm/estimator/log/ckpt \
--summary_save_dir /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/fm/estimator/log/summary \
--batch_size 128 \
--remake_tfrecord
