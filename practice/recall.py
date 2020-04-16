'''
@Author: your name
@Date: 2020-04-15 17:01:35
@LastEditTime: 2020-04-15 17:17:51
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /DeepCTR/practice/recall.py
'''


data_dir = '/Users/liulingzhi5/dataset/dac'
train_dir = data_dir + '/train.txt'
test_dir = data_dir + '/test.txt'

import pandas as pd
# cks = pd.read_csv(train_dir, iterator=True)
# tiny = cks.get_chunk(10000)
# tiny.to_csv('train_tiny.csv', index=False)
tiny = pd.read_csv('practice/train_tiny.csv')