# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 19:39:46 2019

@author: 92156
"""

#没有很好的数据


#标准方程法 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
scale=False




data=pd.read_excel("阴性率与儿童年龄.xls")
x_data=np.array(data["年龄"]).reshape(-1,1)
y_data=np.array(data["阴性率"]).reshape(-1,1)
x_data = np.concatenate((np.ones((7,1)),x_data),axis=1)
def sigmoid(x):
    return 1.0/(1+np.exp(-x_data))
def cost(xMat, yMat, ws):
    left = np.multiply(yMat, np.log(1-sigmoid(xMat*ws)))
    right = np.multiply(1-yMat, np.log(1-sigmoid(xMat * ws)))
    return np.sum(left + right) / -(len(xMat))
def gradDscent(xArr, yArr):
    if scale == True:
        xArr = preprocessing(xArr)
# 不True
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    lr = 0.001    #学习率
    epochs = 10000   #迭代次数
    costList = []
    #计算数据列数，有几列就有几个权值
    m, n = np.shape(xMat)
    #初始化权值
    ws = np.mat(np.ones((n, 1)))
    for i in range(epochs+1):
        # xMat和weights矩阵相乘
        h = sigmoid(xMat*ws)
        # 计算误差
        ws_grad = xMat.T*(h-yMat)/m
        ws = ws - lr * ws_grad
        if i % 50 == 0:
            costList.append(cost(xMat, yMat, ws))
    return ws, costList
ws, costList = gradDscent(x_data, y_data)
print(ws)

if scale == False:
    x_text=[[-4],[3]]
    y_text=(-ws[0]-x_text*ws[1]/ws[2])
    plt.plot(x_text,y_text)
    plt.show()


#sklearn   数据不好
import numpy as np
from sklearn import linear_model, datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report



#https://linjiafengyang.github.io/2018/04/10/%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98%E4%BD%9C%E4%B8%9A%E4%B8%80%EF%BC%9A%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E3%80%81%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92%E5%92%8C%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA/
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_excel("阴性率与儿童年龄.xls")

X = np.array(data["年龄"]).reshape(-1,1)

Y =np.array(data["阴性率"]).reshape(-1,1)

# 逻辑回归
model = LogisticRegression()
model.fit(X, Y)
print("Theta: ", model.coef_)

# 预测
predict = model.predict(X)
right = sum(predict == Y)
print(classification_report(Y,predict))
# 将预测值和真实值放在一块，便于观察
predict = np.hstack((predict.reshape(-1, 1), Y.reshape(-1, 1)))
print("Predict and Y_test: \n", predict)
x_text=np.array([[-4],[3]])
y_text=(-model.intercept_-x_text*model.coef_[0][0])/model.coef_[0][1]
plt.plot(x_text,y_text,"k")
plt.show()

#https://blog.csdn.net/qq_36142114/article/details/80441373
#逻辑回归
import matplotlib.pyplot as plt
import numpy as np


def Sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

x = np.arange(-10, 10, 0.1)
h = Sigmoid(x)  # Sigmoid函数
plt.plot(x, h)
plt.axvline(0.0, color='k')
plt.axhline(y=0.5, ls='dotted', color='k')
plt.yticks([0.0,  0.5, 1.0])  # y axis label
plt.title(r'Sigmoid函数曲线', fontsize = 15)
plt.text(5,0.8,r'$y = \frac{1}{1+e^{-z}}$', fontsize = 18)
plt.show()


#标准方程法   非线性逻辑回归
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
# 定义多项式回归，degree的值可以调节多项式的特征
poly_reg = PolynomialFeatures(degree=3)
# 特征处理
x_poly = poly_reg.fit_transform(x_data)
def sigmoid(x):
    return 1.0/(1+np.exp(-x))


def cost(xMat, yMat, ws):
    left = np.multiply(yMat, np.log(1-sigmoid(xMat*ws)))
    right = np.multiply(1-yMat, np.log(1-sigmoid(xMat * ws)))
    return np.sum(left + right) / -(len(xMat))


def gradDscent(xArr, yArr):
    if scale == True:
        xArr = preprocessing(xArr)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    lr = 0.03
    epochs = 50000
    costList = []
    #计算数据列数，有几列就有几个权值
    m, n = np.shape(xMat)
    #初始化权值
    ws = np.mat(np.ones((n, 1)))
    for i in range(epochs+1):
        # xMat和weights矩阵相乘
        h = sigmoid(xMat*ws)
        # 计算误差
        ws_grad = xMat.T*(h-yMat)/m
        ws = ws - lr * ws_grad
        if i % 50 == 0:
            costList.append(cost(xMat, yMat, ws))
    return ws, costList