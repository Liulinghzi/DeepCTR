<!--
 * @Author: your name
 * @Date: 2020-04-15 13:43:21
 * @LastEditTime: 2020-04-15 14:11:23
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /DeepCTR/视频笔记/dcn.md
 -->


1. w&d的wide部分需要人工进行特征交叉
2. deepfm的wide部分只能进行2阶交叉
3. dcn的wide部分能自动高阶交叉

使用：
根据浅梦评论区的发言，dcn非常容易过拟合，应用价值不大
自己分析非常容易过拟合的原因
cross部分就是简单的多项式连乘，多项式连乘的意义类似于高度笛卡尔积交叉，将整个空间划分成非常细小的部分，由此造成的过拟合

与之对应的deepfm则是embedding之后，求embedding的相似性，不是简单的分割