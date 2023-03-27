# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 16:39:06 2019

@author: 92156
"""

import numpy as np # 快速操作结构数组的工具
import matplotlib.pyplot as plt  # 可视化绘制
import pandas as pd 
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV 
# 样本数据集，第一列为x，第二列为y，在x和y之间建立回归模型


data=pd.read_excel("财政收入.xls")

#生成X和y矩阵
dataMat = np.array(data)
X = dataMat[:,1].reshape(-1,1)  # 变量x
y = dataMat[:,0].reshape(-1,1)   #变量y


# ========Lasso回归========
model = Lasso(alpha=0.01)  # 调节alpha可以实现对拟合的程度
model = LassoCV()  # LassoCV自动调节alpha可以实现选择最佳的alpha。
model = LassoLarsCV()  # LassoLarsCV自动调节alpha可以实现选择最佳的alpha
model.fit(X, y)   # 线性回归建模
print('系数矩阵:\n',model.coef_)
print('线性回归模型:\n',model)
# print('最佳的alpha：',model.alpha_)  # 只有在使用LassoCV、LassoLarsCV时才有效
# 使用模型预测
predicted = model.predict(X)

# 绘制散点图 参数：x横轴 y纵轴
plt.scatter(X, y, marker='x')
plt.plot(X, predicted,c='r')

# 绘制x轴和y轴坐标
plt.xlabel("x")
plt.ylabel("y")

# 显示图形
plt.show()


