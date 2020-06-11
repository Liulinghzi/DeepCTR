'''
@Author: your name
@Date: 2020-06-08 15:30:19
@LastEditTime: 2020-06-08 15:59:35
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /rank/makeseq.py
'''


import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_dir', type=str, default=None)
parser.add_argument('--val_data_dir', type=str, default=None)
parser.add_argument('--user_col', type=str, default=None)
parser.add_argument('--time_col', type=str, default=None)
parser.add_argument('--seq_cols', type=str, default=None)
parser.add_argument('--keep_duplicate_user', type=str, default='False')

args = parser.parse_args()
if args.keep_duplicate_user.lower() == 'false':
    args.keep_duplicate_user = False
elif args.keep_duplicate_user.lower() == 'true':
    args.keep_duplicate_user = True
else:
    raise ValueError('keep_duplicate_user的值错误，只能是True或者False')

behavior_features = [c.strip() for c in args.seq_cols.strip().split(',')]

train = pd.read_csv(args.train_data_dir)
val = pd.read_csv(args.val_data_dir)

concat_data = pd.concat([train, val], axis=1)
concat_data = concat_data.sort_values(by=args.time_col)
concat_data = make_hist_sequences(concat_data, groupby_feature=args.user_col,behavior_features=behavior_features, time_feature=args.time_col, keep_duplicate_user=args.keep_duplicate_user)



def make_hist_sequences(data, groupby_feature, behavior_features, attribute_features, time_feature, max_len, keep_duplicate_user=False):
    """
    Parameters
    ----------
    data: pd.DataFrame
    groupby_feature: str, 计算user的序列还是item的序列，直接传入列名
    behavior_features: 计算哪些行为的序列，[店铺id， 品类id， 商品id]
    attribute_features: 持续不变的特征，一个user_id只有一个
    time_feature: 时间特征列，需要先按照这里进行排序
    max_len: 返回的seq的长度, 不足的用0填充
    """
    # 在numpy层面进行计算，速度快
    if not isinstance(behavior_features, list) or not isinstance(attribute_features, list):
        raise ValueError('behavior_features 和  attribute_features必须为列表')
    else:
        features = [groupby_feature] + behavior_features + attribute_features

    feature_value_list = data[features].sort_values(
        [groupby_feature, time_feature]).values.T

    groupby_feature_value = feature_value_list[0]
    groupby_feature_unique_value, groupby_feature_change_index = np.unique(groupby_feature_value, True)
    # 以groupby_feature作为第一排序，然后直接找到每次groupby_feature变化时候的index

    behavior_features_values = [np.split(values, groupby_feature_change_index[1:]) for values in feature_value_list[1: 1+len(behavior_features)]]    
    attribute_features_values = [values[groupby_feature_change_index] for values in feature_value_list[1+len(behavior_features): ]]

    print(len(feature_value_list))
    print(len(behavior_features_values))
    print(len(attribute_features_values))

    hist_dict = {}
    hist_dict[groupby_feature] = groupby_feature_unique_value
    

    
    for idx, feat in enumerate(behavior_features):
        hist_dict['hist_'+feat] = [list(behavior_seq[:max_len]) if len(behavior_seq)>max_len else list(behavior_seq[:max_len]) + ['0']*(max_len-len(behavior_seq)) for behavior_seq in behavior_features_values[idx]]

    for idx, feat in enumerate(attribute_features):
        hist_dict[feat] = [v for v in attribute_features_values[idx]]

    hist_mapper = pd.DataFrame(hist_dict)
    if not keep_duplicate_user:
        return hist_mapper

    data.set_index(groupby_feature, inplace=True)
    hist_mapper.set_index(groupby_feature, inplace=True)
    data = data.merge(hist_mapper, left_index=True, right_index=True)
    data = data.reset_index()
    # data = data.merge(hist_mapper, on=groupby_feature, how='left')
    return data
