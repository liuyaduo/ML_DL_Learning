# -*- encoding: utf-8 -*-
'''
@File    :   KNN.py
@Time    :   2021/01/14 21:14:08
@Author  :   yaduo 
@Version :   1.0
@Contact :   liuyaduo@outlook.com
'''
from math import sqrt, inf
import numpy as np
class KNNClassifier():
    '''
    K近邻分类器
    待实现：
        1.交叉验证
        2.优化效率：KD树
    '''
    def __init__(self, k_neighbors=5, metric='euclidean', algorithm='base'):
        """
        @description  :
        ---------
        @param  :
        k_neighbors: int, 默认为5
            最近的k个实例，k的大小
        metric:距离度量：str,默认为'euclidean' 
            'manhattan'：曼哈顿距离
            'euclidean'：欧几里得距离
        algorithm: str,默认为'base'  实现KNN所用的算法
            'brute': 暴力算法 O(KDN)
            'KdTree': 使用KD树
        -------
        """
        self.k_neighbors = k_neighbors
        self.metric = metric
        self.algorithm = algorithm

    def _distance(self, x, y):
        if x.shape != y.shape:
            raise ValueError('维度不一致！')
        dim = x.shape[0]
        distance = 0
        if self.metric == 'euclidean':
            for i in range(dim):
                # ^在python中是异或
                distance += (x[i] - y[i]) ** 2
            distance = sqrt(distance) 
        if self.metric == 'manhattan':
            for i in range(dim):
                distance += abs(x[i] - y[i]) 
        return distance

    def kneighbors(self, fit_X, X, k_neighbors=5, return_distance=True):
        """
        @description  :
            返回距离X最近的k的点的索引(和其距离)
        ---------
        @param  :

        -------
        @Returns  : ndarray, [num_queries, 1]
            每一个样本的类别
        -------
        """
        fit_X = np.array(fit_X)
        X = np.array(X)
        fit_X_number = fit_X.shape[0]
        if k_neighbors > fit_X_number:
            raise ValueError('k数值过大！')
        X_number = X.shape[0]
        k_index = np.zeros((X_number, k_neighbors), dtype='int32')
        distances = np.zeros((X_number, k_neighbors), dtype='float64')
        for i, x in enumerate(X):
            distance = np.array([self._distance(x, fit_x) for fit_x in fit_X], dtype='float64')
            for k in range(k_neighbors):
                min_index = np.argmin(distance)
                distances[i][k] = np.min(distance)
                # 将前k个最小的距离放到前面,会使其替换掉原来第i个元素
                k_index[i][k] = min_index
                distance[min_index] = inf

        return k_index if return_distance == False else (distances, k_index)

    def predict(self, fit_X, fit_y, X):
        """
        @description  :
        对测试集数据进行预测。
        ---------
        @param  :
        -------
        @Returns  : list
        -------
        """
        fit_X = np.array(fit_X)
        fit_y = np.array(fit_y)
        X = np.array(X)
        num_queries = X.shape[0]
        num_class = len(set(fit_y))
        neighbors = self.kneighbors(fit_X, X, k_neighbors=self.k_neighbors, return_distance=False)
        #取出样本k个近邻的索引对应的类别
        neighbors_label = fit_y[neighbors]
        pred_y = [0] * num_queries
        # 投票
        for i, query in enumerate(neighbors_label):
            label_list = [0] * num_class
            for label in query:
                label_list[label] += 1
            pred_y[i] = label_list.index(max(label_list))
            
        return pred_y
        
    
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
        

if __name__ == "__main__":
    knn = KNNClassifier(k_neighbors=3)
    fit_X = [[0], [1], [2], [3]]
    fit_y = [0, 0, 1, 1]
    X = [[1.1], [2.2], [2.7]]
    print(knn.predict(fit_X, fit_y, X))
    print(knn.kneighbors(fit_X, X, k_neighbors=3))