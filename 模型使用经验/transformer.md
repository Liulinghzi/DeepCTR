<!--
 * @Author: your name
 * @Date: 2020-05-19 14:07:15
 * @LastEditTime: 2020-05-19 15:54:43
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /DeepCTR/模型使用经验/transformer.md
--> 


1. 关于数据量，如果数据量特别大，一个epoch也许就足够过拟合，这种时候也许需要对数据进行筛选
2. 如果数据量比较小，embeeding可能不是很好学，可以使用预训练的embedding进行初始化，并且固定不训练embedding
    关于模型复杂度，当前的理解为，
        如果模型特别复杂，数据量不是特别大，那么持续训练得到的结果应该是过拟合，而不是无法收敛
            所以需要先用比如1w数据集进行训练，看看到底能不能过拟合
        
3. 