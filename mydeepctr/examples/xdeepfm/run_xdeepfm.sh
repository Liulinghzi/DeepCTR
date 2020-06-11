###
 # @Author: your name
 # @Date: 2020-05-28 18:00:27
 # @LastEditTime: 2020-06-09 18:49:40
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /estimator/run.sh
### 




python3 classfication_train_wrapper.py \
--model xdeepfm \
--data_dir /Users/liulingzhi5/dataset/heart \
--output_path /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/xdeepfm/estimator/log/ckpt_xdeepfm \
--summary_save_dir /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/xdeepfm/estimator/log/summary_xdeepfm \
--target target \
--sparse_cols sex,cp,fbs,restecg,exang,slope,ca,thal \
--num_epoches 1000 \
--batch_size 128 \
--learning_rate 0.001 \
--units 16,16 \
--cin_units 10,10 \
--use_bn False \
--activation relu \
--dropout_rate 0 \
--remake_tfrecord

python3 classfication_val_wrapper.py \
--data_dir /Users/liulingzhi5/dataset/heart \
--input_algor /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/xdeepfm/estimator/log/ckpt_xdeepfm \
--summary_save_dir /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/xdeepfm/estimator/log/summary_xdeepfm \
--batch_size 128 \
--remake_tfrecord




python3 classfication_train_wrapper.py \
--model xdeepfm \
--data_dir /Users/liulingzhi5/dataset/movielens/ml-1m/train.csv \
--tfrecord_dir /Users/liulingzhi5/dataset/movielens/ml-1m/ \
--output_path /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/xdeepfm/estimator/log/ckpt_xdeepfm \
--summary_save_dir /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/xdeepfm/estimator/log/summary_xdeepfm \
--target ratings \
--sparse_cols users,movies,title,genres,gender,occupation \
--num_epoches 100 \
--batch_size 128 \
--learning_rate 0.001 \
--units 10,10 \
--use_bn 1\
--remake_tfrecord

python3 classfication_val_wrapper.py \
--model xdeepfm \
--data_dir /Users/liulingzhi5/dataset/movielens/ml-1m/test.csv \
--tfrecord_dir /Users/liulingzhi5/dataset/movielens/ml-1m/ \
--input_algor /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/xdeepfm/estimator/log/ckpt_xdeepfm \
--summary_save_dir /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/xdeepfm/estimator/log/summary_xdeepfm \
--batch_size 128 \
--remake_tfrecord

