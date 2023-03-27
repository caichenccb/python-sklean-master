# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 12:44:01 2019

@author: 92156
"""


#第一   误差 无穷



import xlrd
import xlwt
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
workbook=np.array(pd.read_excel("财政收入.xls"))

cols1=workbook[:,1]   #获取第一列
cols2=workbook[:,0]   #获取第二列

#a+bx
#a=sp.Symbol('a')
#b=sp.Symbol('b')
#已知a=-22.63450339669057 b=13.449314363947979

def get_sqm(a,b):
    sqm=0
    for i in range(len(cols1)):
        sqm=sqm+(cols2[i]-a-b*cols1[i])*(cols2[i]-a-b*cols1[i])
    return sqm

def get_pa(a,b):
    pa=0
    for i in range(len(cols1)):
        pa=pa-2*(cols2[i]-a-b*cols1[i])
    return pa

def get_pb(a,b):
    pb=0
    for i in range(len(cols1)):
        pb=pb-2*cols1[i]*(cols2[i]-a-b*cols1[i])
    return pb
n=0.00001
a=0.0
b=0.0
while abs(get_pa(a,b))>=1 and abs(get_pb(a,b))>=1 :
    c=a
    d=b
    a=a-n*get_pa(c,d)
    b=b-n*get_pb(c,d)
    print(get_sqm(a,b))

print(a,b)
plt.scatter(cols1,cols2,color = 'blue')
x=np.linspace(0,0.01,len(cols1))
y=b*x+a
plt.plot(x,y,color="red")
plt.show()



#第二   死循环


import numpy as np
import pandas as pd

# Size of the points dataset.
m = 22
workbook=np.array(pd.read_excel("财政收入.xls"))

X1=workbook[:,0].reshape(m,1)   #获取第一列
y=workbook[:,1]   #获取第二列

# Points x-coordinate and dummy value (x0, x1).
X0 = np.ones((m, 1))
X = np.hstack((X0, X1))

# Points y-coordinate

# The Learning Rate alpha.
alpha = 0.01

def error_function(theta, X, y):
    '''Error function J definition.'''
    diff = np.dot(X, theta) - y
    return (1./2*m) * np.dot(np.transpose(diff), diff)

def gradient_function(theta, X, y):
    '''Gradient of the function J definition.'''
    diff = np.dot(X, theta) - y
    return (1./m) * np.dot(np.transpose(X), diff)

def gradient_descent(X, y, alpha):
    '''Perform gradient descent.'''
    theta = np.array([1, 1]).reshape(2, 1)
    gradient = gradient_function(theta, X, y)
    while not np.all(np.absolute(gradient) <= 1e-5):
        theta = theta - alpha * gradient
        gradient = gradient_function(theta, X, y)
    return theta

optimal = gradient_descent(X, y, alpha)
print('optimal:', optimal)
print('error function:', error_function(optimal, X, y)[0,0])

#第三

# 训练集
# 每个样本点有3个分量 (x0,x1,x2)
data=np.array(pd.read_excel("财政收入.xls"))
x=data[:,1:]
y=data[:,0]
#x = [(1, 0., 3), (1, 1., 3), (1, 2., 3), (1, 3., 2), (1, 4., 4)]
## y[i] 样本点对应的输出
#y = [95.364, 97.217205, 75.195834, 60.105519, 49.342380]
 
# 迭代阀值，当两次迭代损失函数之差小于该阀值时停止迭代
epsilon = 0.0001
 
# 学习率
alpha = 0.01
diff = [0, 0]
max_itor = 1000
error1 = 0
error0 = 0
cnt = 0
m = len(x)
 
 
# 初始化参数
theta0 = 0
theta1 = 0
theta2 = 0
 
while True:
    cnt += 1
 
    # 参数迭代计算
    for i in range(m):
        # 拟合函数为 y = theta0 * x[0] + theta1 * x[1] +theta2 * x[2]
        # 计算残差
        diff[0] = (theta0 + theta1 * x[i][1] + theta2 * x[i][2]) - y[i]
 
        # 梯度 = diff[0] * x[i][j]
        theta0 -= alpha * diff[0] * x[i][0]
        theta1 -= alpha * diff[0] * x[i][1]
        theta2 -= alpha * diff[0] * x[i][2]
 
    # 计算损失函数
    error1 = 0
    for lp in range(len(x)):
        error1 += (y[lp]-(theta0 + theta1 * x[lp][1] + theta2 * x[lp][2]))**2/2
 
    if abs(error1-error0) < epsilon:
        break
    else:
        error0 = error1
 
    print (' theta0 : %f, theta1 : %f, theta2 : %f, error1 : %f' % (theta0, theta1, theta2, error1))
print ('Done: theta0 : %f, theta1 : %f, theta2 : %f' % (theta0, theta1, theta2))
print ('迭代次数: %d' % cnt)

#第四次
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
#模拟数据
data=np.array(pd.read_excel("财政收入.xls"))
x=data[:,1]
y=data[:,0] 
#计算损失函数
def compute_cost(x, y, theta):
    y_pred = np.dot(x, theta.T)
    inner = np.power((y_pred-y), 2)
    cost = np.sum(inner, axis=0) / (2 * x.shape[0])
    return cost
#梯度下降
def grandient_descent(x, y, theta, alpha, iters):
    #参数长度
    len_params = theta.shape[1]
    #参数更新次数
    for it in range(iters):
        error_val = np.dot(x, theta.T) - y
        error_val = np.reshape(error_val, (22,))
        update_val_temp = np.zeros(shape=x.shape)
        #参数个数
        for p in range(len_params):
            update_val_temp[:,p] = error_val * x[:,p]
        update_val = np.mean(update_val_temp, axis=0)
        #更新参数
        theta = theta - alpha * update_val
        print('第%d次训练===截距：%f，斜率%f' % (it, theta[0,0], theta[0,1]))
    cost = compute_cost(x, y, theta)
    return cost, theta

#初始化参数
#x插入一列值为1
x = np.reshape(x, (-1,1))
x = np.insert(x, 0, 1, axis=1)
#y值重新reshape一下
y = np.reshape(y, (-1,1))
theta = np.zeros(shape=(1, x.shape[1]))
#初始化参数
alpha = 0.01
iters = 2000
cost, theta_v = grandient_descent(x, y, theta, alpha, iters)
print(cost, theta_v)

#第五次
def run_gradient_descent(x_data, y_data, b, w, learn_rate, epochs):
    """
    运行梯度下降
    :param x_data: 待训练的特征数据
    :param y_data: 标签数据
    :param b: 截距
    :param w1: 斜率1
    :param w2: 斜率2
    :param learn_rate: 学习曲率
    :param epochs: 训练次数
    """
    
    m = float(len(x_data))
    for i in range(epochs):
        b_grad = 0 # 损失（代价）函数对b的梯度
        w1_grad = 0 # 损失（代价）函数对w1的梯度
        w2_grad = 0 # 损失（代价）函数对w2的梯度
        for j in range(0, len(x_data)):
            b_grad += (1/m) * ((b + w1 * x_data[j, 0] + w2 * x_data[j, 1]) - y_data[j])
            w1_grad += (1/m) * ((b + w1 * x_data[j, 0] + w2 * x_data[j, 1]) - y_data[j]) * x_data[j, 0]
            w2_grad += (1/m) * ((b + w1 * x_data[j, 0] + w2 * x_data[j, 1]) - y_data[j]) * x_data[j, 1]
        # 根据梯度和学习曲率修正截距b和斜率w1、w2
        b -= learn_rate * b_grad
        w1 -= learn_rate * w1_grad
        w2 -= learn_rate * w2_grad
        if i % 100 == 0:
        	# 每100次作图一次
            print("epochs:", i)
            ax = plt.figure().add_subplot(111, projection = "3d")
            ax.scatter(x_data[:, 0], x_data[:, 1], y_data, c = "b", marker = 'o', s = 10)
            x0 = x_data[:, 0]
            x1 = x_data[:, 1]
            x0, x1 = np.meshgrid(x0, x1)
            z = b + w1 * x0 + w2 * x1
            ax.plot_surface(x0, x1, z, color = "r")
            ax.set_xlabel("area")
            ax.set_ylabel("num_rooms")
            ax.set_zlabel("price")
            plt.show()
            print("mse: ", compute_mse(b, w1, w2, x_data, y_data))
            print("------------------------------------------------------------------------------------------------------------")
    return b, w1, w2

learn_rate = 0.0001
b = 0
w1 = 0
w2 = 0
epochs = 1000

print("Start args: b = {0}, w1 = {1}, w2 = {2}, mse = {3}".format(b, w1, w2, compute_mse(b, w1, w2, x_data, y_data)))
print("Running...")
b, w1, w2 = run_gradient_descent(x_data, y_data, b, w1, w2, learn_rate, epochs)
print("Finish args: iterations = {0}  b = {1}, w1 = {2}, w2 = {3}, mse = {4}".format(epochs, b, w1, w2, compute_mse(b, w1, w2, x_data, y_data)))