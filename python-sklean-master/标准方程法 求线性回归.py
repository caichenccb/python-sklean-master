# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 22:58:35 2019

@author: 92156
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
data=pd.read_excel("财政收入.xls")
x_data =np.array(data["x1"])
y_data =np.array( data["y"])
# 作图
plt.scatter(x_data, y_data)
plt.show()
#. 数据预处理
x_data = x_data[:, np.newaxis]
y_data = y_data[:, np.newaxis]
# 添加偏置
X_data = np.concatenate((np.ones((22, 1)), x_data), axis = 1)
print(X_data)
def calc_weights(X_data, y_data):
    """
    标准方程法求weights
    算法: weights = (X的转置矩阵 * X矩阵)的逆矩阵 * X的转置矩阵 * Y矩阵
    :param x_data: 特征数据
    :param y_data: 标签数据
    """
    
    x_mat = np.mat(X_data)
    y_mat = np.mat(y_data)
    xT_x = x_mat.T * x_mat
    if np.linalg.det(xT_x) == 0:
        print("x_mat为不可逆矩阵，不能使用标准方程法求解")
        return
    weights = xT_x.I * x_mat.T * y_mat
    return weights
weights = calc_weights(X_data, y_data)
print(weights)
x_test = np.array(np.array(data["x1"]))[:, np.newaxis]
predict_result = weights[0] + x_test * weights[1]
# 原始数据
plt.plot(x_data, y_data, "b.")
# 预测数据
plt.plot(x_test, predict_result, "r")
plt.show()