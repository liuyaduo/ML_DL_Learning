# -*- encoding: utf-8 -*-
'''
@File    :   KNN.py
@Time    :   2021/01/14 21:14:08
@Author  :   yaduo 
@Version :   1.0
@Contact :   liuyaduo@outlook.com
'''

class KNNClassifier():
    '''
    K近邻分类器
    待实现：
        1.交叉验证
        2.优化效率：KD树
    '''
    def __init__(self, k_neighbors=5, metric='euclidean'):
        """
        @description  :
        ---------
        @param  :
        k_neighbors: int, 默认为5
            最近的k个实例，k的大小
        metric:距离度量：str,默认为'euclidean' 
            'manhattan'：曼哈顿距离
            'euclidean'：欧几里得距离
        -------
        @Returns  :
        -------
        """
    
    def predict(self):
        """
        @description  :
        对测试集数据进行预测。
        ---------
        @param  :
        -------
        @Returns  :
        -------
        """
    
    def predict_prob(self):
        """
        @description  :
        得到测试集上的正确率
        ---------
        @param  :
        -------
        @Returns  :
        -------
        """
        
        
    