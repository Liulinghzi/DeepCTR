'''
@Author: your name
@Date: 2020-05-27 10:52:57
@LastEditTime: 2020-06-09 18:26:42
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /model-building/recommend/xdeepfm/xdeepfm.py
'''
import pandas as pd
import tensorflow as tf
from inputs import SparseFeature, DenseFeature, build_input_placeholder, build_embedding_matrix_dict, input_from_feature_columns
from layers.baselayers import Linear, DNN
import six
import copy


class XDeepFMConfig():
    def __init__(self, dnn_feature_columns, linear_feature_columns, class_num, units, activation, dropout_rate, use_bn, cin_units):
        self.dnn_feature_columns = dnn_feature_columns
        self.linear_feature_columns = linear_feature_columns
        self.class_num = class_num
        self.units = units
        self.cin_units = cin_units
        self.use_bn = use_bn
        self.activation = activation
        self.dropout_rate = dropout_rate

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = XDeepFMConfig(vocab_size=None)
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


class XDeepFM():
    # 当前理解(2020年06月01日 星期一)：
    # xdeepfm
    # 动机，DCN没有域的概念，按理来说，一个特征内部embeding的不同维度，不应该交互，但是DCN做了交互
    # XdeepFM引入域，避免这一步操作
    # NOTE:注意XdeepFM中域的概念和FFM中域的概念并不一样，XdeepFM中的域只是vector-wise, FFM中是吧几个特征分组

    # DCN中，把n个feature concat成了[bs, num_feature * dim]的维度
    # CIN中， 把n个feature concat成了 [bs, num_feature, dim]的维度
    # 抛开系数矩阵不看，从根本上看CIN的目的就是交互不同的特征,即 m 个特征和 m个特征的 m * m 种交互
    # 而为了给 m * m种交互加上系数， 添加了 hk * m * m, 并且划分为(hk * m) * m
    # 其中(hk * m)就是CIN中的一层， [hk * D] [m * D] => [hk * m * D], 并且保持m * D形状不变，沿着hk轴求mean pooling
                                    # 这个计算，按理来说是对位相乘，但是这里并不对位，而是笛卡尔积交叉hk和m， 然后在对D进行对位相乘
                                    # 所以这里可以stack成[hk * m * D]的形状，然后开始对位相乘

    
    # 理解更新(2020年06月03日 星期三)

    # 其实CIN的交互方式和DCN的交互方式基本完全相同，只是bit-wise和vector-wise的区别
    # CIN
    #   m个特征，两两交互，得到[m, m]种交互，每种交互还有D的维度，然后对交互结果进行hk-1种加权, 得到 hk-1个加权结果，每个结果D维，所以是[hk-1, D]
    # DCN
    #   m*dim个维度，两两交互，得到[m*dim, m*dim]种交互,每种交互只有1维, 然后对交互结果进行稳定的[m*dim]种加权， 得到m*dim种加权结果， 每个结果1位，所谓是[m*dim, 1]


    # 除了交互方式完全相同之外，还有一点细微差别
    # CIN
    #   每次交互完了之后，不加上原始输入，那么下次交互，不会生成2阶特征，所以第l层只有l阶的交互形式
    #   造成结果 ===》需要每一个交互层的输出结果
    #       每一层的交互结果沿着D维度sum pool，并且拼接
    #       TODO:理论依据：等效FM，有待理解
    # DCN
    #   每次交互完了之后，还要加上原始输入，那么下次交互，还会生成2阶特征，所以第l层有1~l阶所有的交互形式
    #   造成结果 ===》直接使用最后一个交互层的输出结果


    # 其中(hk * m)就是CIN中的一层， [hk * D] [m * D] => [hk * m * D], 其中hk中的一行就是对m个特征的一个权重向量，对m轴进行reduce_sum，
    # 得到[hk, D],其中hk中的每一个维度，是m个特征的一种线性组合，   那么[hk, D]再和[m, D]交互， 就从线性组合，变成了二阶的组合， 成为[hk, m, D]
    # 然后采用hk+1种权重矩阵对成为[hk, m, D]进行加和    tensordot([hk+1, hk, m], [hk, m, d]  , axes=2) ==> hk+1, d


    # https://zhuanlan.zhihu.com/p/83784018
    # 应该是用cnn对特征进行提取？？？还是deep&cross
    def __init__(self, model_config, inputs, labels, scope='XDeepFM', mode=tf.estimator.ModeKeys.TRAIN):
        self.config = copy.deepcopy(model_config)
        self.inputs = inputs
        self.labels = labels
        self.mode = mode

        with tf.variable_scope(scope, default_name='embeddings'):
            linear_dense_value_list, linear_sparse_embedding_list = input_from_feature_columns(
                self.inputs, self.config.linear_feature_columns, target='linear')
            dnn_dense_value_list, dnn_sparse_embedding_list = input_from_feature_columns(
                self.inputs, self.config.dnn_feature_columns, target='dnn')

        linear_logits = Linear()(linear_dense_value_list, linear_sparse_embedding_list)
        dnn_logits = DNN(units=self.config.units, activation=self.config.activation, dropout_rate=self.config.dropout_rate, use_bn=self.config.use_bn, training=self.mode==tf.estimator.ModeKeys.TRAIN)(dnn_dense_value_list, dnn_sparse_embedding_list)

        self.logits = linear_logits + dnn_logits
        if len(dnn_sparse_embedding_list) > 0:
            cin_logits = self.cin(dnn_sparse_embedding_list)        

            self.logits += cin_logits

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            self.loss = None
        else:
            if len(labels.shape) == 1:
                labels = tf.expand_dims(labels, axis=-1)
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits, labels=labels) + tf.losses.get_regularization_loss())


    def cin(self, embedding_list):
        with tf.variable_scope('cin'):
            # 由于tfrecord里面的变量全都是list，所以这里的embedding lookup的结果维度应该是[bs, 1, dim]  错误，实际上是[bs, dim]
            # 可能并不是list，
            concat_sparse = tf.stack(embedding_list, axis=1)
            output_list = []
            xi = concat_sparse # bs, m, dim
            for idx, unit in enumerate(self.config.cin_units):
                # xi 和 和 concat_sparse进行交互 =》 bs, m, m, dim
                new_shape = [-1, xi.shape[1], concat_sparse.shape[1], xi.shape[-1]]
                stack_xi = tf.stack([xi] * new_shape[2], axis=2) # [bs, hk-1, m, dim]
                stack_concat_sparse = tf.stack([concat_sparse] * new_shape[1], axis=1) # [bs, hk-1, m, dim]
                dot_ = tf.multiply(stack_xi, stack_concat_sparse) # [bs, hk-1, m, dim]

                cross_weight = tf.get_variable(name='cross_weight_%d' % idx, shape=[unit, new_shape[1], new_shape[2]]) #   unit, dim
                xi = tf.tensordot(dot_, cross_weight, axes=((1,2), (1,2)))
                xi = tf.transpose(xi, perm=(0,2,1))
                output_list.append(tf.reduce_sum(xi, axis=-1))

            cin_logits = tf.layers.dense(tf.concat(output_list, axis=-1), 1, use_bias=True, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
            
            return cin_logits


        
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

# xdeepfm = XDeepFM(sparse_feature_columns, dense_feature_columns, input_placeholders)
# data = pd.DataFrame()
# xdeepfm.fit(feed_dict, data[target])
# xdeepfm.transform(data)
