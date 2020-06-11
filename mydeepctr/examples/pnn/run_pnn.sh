###
 # @Author: your name
 # @Date: 2020-05-28 18:00:27
 # @LastEditTime: 2020-06-10 10:42:08
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /estimator/run.sh
### 
# age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target

python3 classfication_train_wrapper.py \
--model pnn \
--data_dir /Users/liulingzhi5/dataset/movielens/ml-1m/train.csv \
--tfrecord_dir /Users/liulingzhi5/dataset/movielens/ml-1m/ \
--output_path /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/pnn/estimator/log/ckpt/pnn \
--summary_save_dir /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/pnn/estimator/log/summary/pnn \
--target ratings \
--sparse_cols users,movies,title,genres,gender,occupation \
--num_epoches 100 \
--batch_size 128 \
--use_bn False \
--dropout_rate 0 \
--activation relu \
--learning_rate 0.001 \
--remake_tfrecord


python3 classfication_val_wrapper.py \
--model dcn \
--data_dir /Users/liulingzhi5/dataset/movielens/ml-1m/test.csv \
--tfrecord_dir /Users/liulingzhi5/dataset/movielens/ml-1m/ \
--input_algor /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/dcn/estimator/log/ckpt/dcn \
--summary_save_dir /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/dcn/estimator/log/summary/dcn \
--batch_size 128 \
--remake_tfrecord



# python3 classfication_train_eval_wrapper.py \
# --model dcn \
# --data_dir /Users/liulingzhi5/dataset/movielens/ml-1m/train.csv \
# --tfrecord_dir /Users/liulingzhi5/dataset/movielens/ml-1m/ \
# --output_path /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/dcn/estimator/log/ckpt/dcn \
# --summary_save_dir /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/dcn/estimator/log/summary/dcn \
# --target ratings \
# --sparse_cols users,movies,title,genres,gender,occupation \
# --num_epoches 100 \
# --batch_size 128 \
# --learning_rate 0.001 \
# --num_crosses 2 \
# --remake_tfrecord


python3 classfication_train_wrapper.py \
--model pnn \
--data_dir /Users/liulingzhi5/dataset/heart \
--output_path /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/pnn/estimator/log/ckpt/pnn \
--summary_save_dir /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/pnn/estimator/log/summary/pnn \
--target target \
--sparse_cols sex,cp,fbs,restecg,exang,slope,ca,thal \
--num_epoches 1000 \
--batch_size 128 \
--learning_rate 0.001 \
--method inner \
--units 16,16 \
--activation relu \
--dropout_rate 0 \
--use_bn False \
--remake_tfrecord

python3 classfication_val_wrapper.py \
--data_dir /Users/liulingzhi5/dataset/heart \
--input_algor /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/pnn/estimator/log/ckpt/pnn \
--summary_save_dir /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/pnn/estimator/log/summary/pnn \
--batch_size 128 \
--remake_tfrecord


