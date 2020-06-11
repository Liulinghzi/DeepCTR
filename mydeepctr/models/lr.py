'''
@Author: your name
@Date: 2020-05-27 10:52:57
@LastEditTime: 2020-06-09 13:07:14
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /model-building/recommend/lr/lr.py
'''
import pandas as pd
import tensorflow as tf
from inputs import SparseFeature, DenseFeature, build_input_placeholder, build_embedding_matrix_dict, input_from_feature_columns
from layers.baselayers import Linear, DNN
import six
import copy


class LRConfig():
    def __init__(self, dnn_feature_columns, linear_feature_columns, class_num):
        self.dnn_feature_columns = dnn_feature_columns
        self.linear_feature_columns = linear_feature_columns
        self.class_num = class_num

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = LRConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class LR():
    def __init__(self, model_config, inputs, labels, scope='LR', mode=tf.estimator.ModeKeys.TRAIN):
        self.config = copy.deepcopy(model_config)
        self.inputs = inputs
        self.labels = labels
        self.mode = mode

        with tf.variable_scope(scope, default_name='embeddings'):
            linear_dense_value_list, linear_sparse_embedding_list = input_from_feature_columns(
                self.inputs, self.config.linear_feature_columns, target='linear')


        linear_logits = Linear()(linear_dense_value_list, linear_sparse_embedding_list)
        self.logits = linear_logits

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            self.loss = None
        else:
            if len(labels.shape) == 1:
                labels = tf.expand_dims(labels, axis=-1)
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits, labels=labels) + tf.losses.get_regularization_loss())


    def lr(self, embedding_list):
        with tf.variable_scope('lr'):
            # 由于tfrecord里面的变量全都是list，所以这里的embedding lookup的结果维度应该是[bs, 1, dim]  错误，实际上是[bs, dim]
            # 可能并不是list，
            lr_input = tf.stack(embedding_list, axis=1)
            # stack堆叠会多出一维， [bs, num_features, dim]

            square_of_sum = tf.square(tf.reduce_sum(lr_input, axis=1))
            sum_of_square = tf.reduce_sum(tf.square(lr_input), axis=1)
            lr_logits = 0.5 * tf.reduce_sum(square_of_sum - sum_of_square, axis=1)
            lr_logits = tf.expand_dims(lr_logits, axis=-1)
            return lr_logits

    def get_logits(self):
        return self.logits

    def get_loss(self):
        return self.loss


feed_dict = {}


# parser = argparse.ArgumentParser()
# parser.add_argument("--data_dir", type=str, default="../data/")
# parser.add_argument("--output_path", type=str, default="./output/")
# parser.add_argument("--sparse_features", type=str, default="virginica")
# parser.add_argument("--embedding_dim", type=int, default=4)
# parser.add_argument("--target", type=str, default="virginica")
# parser.add_argument("--exclude", type=str, default="#")
# parser.add_argument("--n_estimators", type=int, default=100)
# args = parser.parse_args()


# data = pd.read_csv(args.data_dir)

# sparse_features = [feat.strip() for feat in args.sparse_features.split(',')]
# dense_features = [feat for feat in data.columns if feat not in sparse_features and feat not != args.target]
# target = args.target

# sparse_feature_columns = [SparseFeature(feature_name=feat, embedding_dim=args.embedding_dim) for feat in sparse_features]
# dense_feature_columns = [DenseFeature(feature_name=feat) for feat in dense_features]
# feature_columns = sparse_feature_columns + dense_feature_columns
# input_placeholders = build_input_placeholder(feature_columns)

# feed_dict = {feature:data[feature] for feature in input_placeholders}
# # 为什么要把placeholders放到类的外面来声明：因为需要用类名构建feed_dict
# #   其实为了保持类的封闭性，是可以把placeholder的构建放到类的里面的，并暴露一个get_placeholders的接口，返回类中构建的placeholder，构建feed_dict

# lr = LR(sparse_feature_columns, dense_feature_columns, input_placeholders)
# data = pd.DataFrame()
# lr.fit(feed_dict, data[target])
# lr.transform(data)
