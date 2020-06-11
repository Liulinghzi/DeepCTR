###
 # @Author: your name
 # @Date: 2020-05-28 18:00:27
 # @LastEditTime: 2020-06-09 18:50:02
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /estimator/run.sh
### 
python3 classfication_train_wrapper.py \
--model wdl \
--data_dir /Users/liulingzhi5/dataset/heart \
--output_path /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/wdl/estimator/log/ckpt \
--summary_save_dir /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/wdl/estimator/log/summary \
--target target \
--sparse_cols sex,cp,fbs,restecg,exang,slope,ca,thal \
--num_epoches 10000 \
--batch_size 128 \
--learning_rate 0.001 \
--activation relu \
--dropout_rate 0 \
--use_bn False \
--units 16,16 \
--remake_tfrecord

python3 classfication_val_wrapper.py \
--data_dir /Users/liulingzhi5/dataset/heart \
--input_algor /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/wdl/estimator/log/ckpt \
--summary_save_dir /Users/liulingzhi5/Desktop/project/组件封装/model-building/recommend/rank/wdl/estimator/log/summary \
--batch_size 128 \
--remake_tfrecord




