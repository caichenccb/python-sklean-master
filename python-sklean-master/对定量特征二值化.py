# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 23:30:01 2019

@author: 92156
"""

from sklearn.preprocessing import Binarizer
 
#二值化，阈值设置为3，返回值为二值化后的数据
Binarizer(threshold=3).fit_transform(iris.data)
#输出1和0