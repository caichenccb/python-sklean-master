# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 23:28:28 2019

@author: 92156
"""


import numpy as np
from sklearn import preprocessing
x=np.array([[1.,-1.,2.],
            [2.,0.,0.],
            [0.,1.,-1.]])
x_normalized=preprocessing.normalize(x,norm='l2')
print(x_normalized)


# 可以使用processing.Normalizer()类实现对训练集和测试集的拟合和转换
normalizer=preprocessing.Normalizer().fit(x)
print(normalizer)
normalizer.transform(x)