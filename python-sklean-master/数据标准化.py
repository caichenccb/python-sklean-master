# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 23:25:48 2019

@author: 92156
"""

from sklearn.preprocessing import StandardScaler
 
#标准化，返回值为标准化后的数据
StandardScaler().fit_transform(iris.data)