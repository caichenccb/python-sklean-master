# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 23:30:35 2019

@author: 92156
"""

from sklearn.preprocessing import OneHotEncoder
 
#哑编码，对IRIS数据集的目标值，返回值为哑编码后的数据
OneHotEncoder().fit_transform(iris.target.reshape((-1,1)))