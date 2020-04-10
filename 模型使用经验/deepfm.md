<!--
 * @Author: your name
 * @Date: 2020-04-09 21:48:04
 * @LastEditTime: 2020-04-09 21:54:23
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /DeepCTR/模型使用经验/deepfm.md
 -->


# 数据预处理
1. 数值特征
    1. 归一化后直接输入dnn部分，不参与fm部分交叉（原生deepfm）
        
        a. 可以考虑分箱后再归一化

    2. 不归一化经过bn后输入dnn，不参与fm部分交叉

    3. 离散化后作为id feature, embedding后与其他sparse feature的embedding 一起参与fm的交叉
    
    4. 为每一个field下的dense value x维护一个embedding vector v,取x·v作为其最终的embedding表示，与其他sparse feature的embedding一起参与fm的交叉（极端化后为原生fm）
    
    3和4的区别在于3中根据dense value的取值会分配到不同的embedding vector，而4中的不同的dense value只有一个embedding vector

