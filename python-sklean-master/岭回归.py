# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 14:37:14 2019

@author: 92156
"""

#调用sklean

#https://blog.csdn.net/luanpeng825485697/article/details/79829778

import numpy as np # 快速操作结构数组的工具
import matplotlib.pyplot as plt  # 可视化绘制
from sklearn.linear_model import Ridge,RidgeCV   # Ridge岭回归,RidgeCV带有广义交叉验证的岭回归
import pandas as pd
from sklearn.metrics import mean_squared_error

# 样本数据集，第一列为x，第二列为y，在x和y之间建立回归模型
data=pd.read_excel("财政收入.xls")


#生成X和y矩阵
dataMat = np.array(data)
X = dataMat[:,1:]  # 变量x
y = dataMat[:,0]  #变量y


# ========岭回归========
alphas = 10**np.linspace(- 3, 3, 100)
model = Ridge(alpha=alphas)
model = RidgeCV(alphas=alphas,store_cv_values=True)  # 通过RidgeCV可以设置多个参数值，算法使用交叉验证获取最佳参数值
model.fit(X, y)   # 线性回归建模
print(model.alpha_)#岭系数
print(model.cv_values_.shape)#loss值
plt.plot(alphas,model.cv_values_.mean(axis=0))
plt.plot(model.alpha_,min(model.cv_values_.mean(axis=0)),"bo")
print(model.coef_)  # 系数
print(model.intercept_)  # 常量
print(model.score(X, y))  # R^2，拟合优度
print(model.get_params())  # 获取参数信息
print(model.set_params(fit_intercept=False))  # 重新设置参数
# print('交叉验证最佳alpha值',model.alpha_)  # 只有在使用RidgeCV算法时才有效
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
RMSE = np.sqrt(mean_squared_error(X,predicted))
RMSE
#发现得到的RMSE更小，说明LASSO回归模型的拟合效果会更贴近于Hitters数据集的原貌。

#标准方程法 岭回归
import numpy as np # 快速操作结构数组的工具
import matplotlib.pyplot as plt  # 可视化绘制
from sklearn.linear_model import Ridge,RidgeCV   # Ridge岭回归,RidgeCV带有广义交叉验证的岭回归
import pandas as pd
from sklearn.metrics import mean_squared_error


data=pd.read_excel("财政收入.xls")


#生成X和y矩阵
dataMat = np.array(data)
x_data= dataMat[:,1].reshape(-1,1)  # 变量x
y_data = dataMat[:,0].reshape(-1,1)  #变量y
X_data = np.concatenate((np.ones((22,1)),x_data),axis=1)
def weights(xArr, yArr, lam=0.2):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    xTx = xMat.T*xMat # 矩阵乘法
    rxTx = xTx + np.eye(xMat.shape[1])*lam
    # 计算矩阵的值,如果值为0，说明该矩阵没有逆矩阵
    if np.linalg.det(rxTx) == 0.0:
        print("This matrix cannot do inverse")
        return
    # xTx.I为xTx的逆矩阵
    ws = rxTx.I*xMat.T*yMat
    return ws

ws = weights(X_data,y_data)
print(ws)

# 计算预测值
aa=np.mat(X_data)*np.mat(ws)
plt.plot(aa)
