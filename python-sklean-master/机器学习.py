# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 10:26:33 2019

@author: 92156
"""

#https://edu.51cto.com/center/course/lesson/index?id=349298&player=h5
#机器学习

#一元线性回归 梯度下降法                
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
# 载入数据
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
data=np.array(pd.read_excel("财政收入.xls"))
x_data=data[:,1]
y_data=data[:,0]
#plt.scatter(x_data,y_data)
#plt.show()

# 学习率learning rate
lr = 0.000000001
# 截距
b = 0 
# 斜率
k = 0 
# 最大迭代次数
epochs = 10000

# 最小二乘法
def compute_error(b, k, x_data, y_data):
    totalError = 0
    for i in range(0, len(x_data)):
        totalError += (y_data[i] - (k * x_data[i] + b)) ** 2
    return totalError / float(len(x_data)) / 2.0

def gradient_descent_runner(x_data, y_data, b, k, lr, epochs):
    # 计算总数据量
    m = float(len(x_data))
    # 循环epochs次
    for i in range(epochs):
        b_grad = 0
        k_grad = 0
        # 计算梯度的总和再求平均
        for j in range(0, len(x_data)):
            b_grad += (1/m) * (((k * x_data[j]) + b) - y_data[j])
            k_grad += (1/m) * x_data[j] * (((k * x_data[j]) + b) - y_data[j])
        # 更新b和k
        b = b - (lr * b_grad)
        k = k - (lr * k_grad)
        # 每迭代5次，输出一次图像
#         if i % 5==0:
#             print("epochs:",i)
#             plt.plot(x_data, y_data, 'b.')
#             plt.plot(x_data, k*x_data + b, 'r')
#             plt.show()
    return b, k

print("Starting b = {0}, k = {1}, error = {2}".format(b, k, compute_error(b, k, x_data, y_data)))
print("Running...")
b, k = gradient_descent_runner(x_data, y_data, b, k, lr, epochs)
print(k,b)
print("After {0} iterations b = {1}, k = {2}, error = {3}".format(epochs, b, k, compute_error(b, k, x_data, y_data)))

# 画图
#b. b是blue  .是画点
plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, k*x_data + b, 'r')
plt.show()    

#调用scikit learn库实现一元线性回归
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# 载入数据
data=np.array(pd.read_excel("财政收入.xls"))
x_data=data[:,1]
y_data=data[:,0]
#plt.scatter(x_data,y_data)
#plt.show()
#print(x_data.shape) 
#/*输出结果为(100,)，是一维数组的意思，但是sklearn需要接受二维数组，
#	所以要将一维数组转成二维数组，即下面两行代码
#*/
x_data = data[:,0,np.newaxis]
y_data = data[:,1,np.newaxis]
# 创建并拟合模型
model = LinearRegression()
model.fit(x_data, y_data)

# 画图
plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, model.predict(x_data), 'r')
plt.show()


#多元线性回归 梯度下降法
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 载入数据
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
data=np.array(pd.read_excel("财政收入.xls"))
x_data=data[:,1:]
y_data=data[:,0]            
# 学习率learning rate
lr = 0.00000000007
#参数
theta0=0
theta1=1
theta2=2
# 最大迭代次数
epochs = 10000

# 最小二乘法
def compute_error(theta0, theta1, theta2,x_data, y_data):
    totalError = 0
    for i in range(0, len(x_data)):
        totalError += (y_data[i] - (theta1 * x_data[i,0] + theta2*x_data[i,1]+theta0)) ** 2
    return totalError / float(len(x_data)) 
def gradient_descent_runner(x_data, y_data, theta0, theta1,theta2, lr, epochs):
    # 计算总数据量
    m = float(len(x_data))
    # 循环epochs次
    for i in range(epochs):
        theta0_grad = 0
        theta1_grad = 0
        theta2_grad=0
        # 计算梯度的总和再求平均
        for j in range(0, len(x_data)):
            theta0_grad += (1/m) * (((theta1 * x_data[j,0]) + theta0+theta2*x_data[j,1]) - y_data[j])
            theta1_grad += (1/m) * x_data[j,0]*((theta1 * x_data[j,0] + theta0+theta2*x_data[j,1]) - y_data[j])
            theta2_grad+=(1/m)*x_data[j,1]*(((theta1 * x_data[j,0]) + theta0+theta2*x_data[j,1]) - y_data[j])
        # 更新
        theta0=theta0-(lr*theta0_grad)
        theta1=theta1-(lr*theta1_grad)
        theta2=theta2-(lr*theta2_grad)
        # 每迭代5次，输出一次图像
#         if i % 5==0:
#             print("epochs:",i)
#             plt.plot(x_data, y_data, 'b.')
#             plt.plot(x_data, k*x_data + b, 'r')
#             plt.show()
    return theta0,theta1,theta2

print("Starting theta0 = {0}, theta1 = {1},theta2={2}, error = {3}".format(theta0, theta1,theta2, compute_error(theta0, theta1,theta2, x_data, y_data)))
print("Running...")
theta0,theta1,theta2= gradient_descent_runner(x_data, y_data, theta0, theta1,theta2, lr, epochs)
print(k,b)
print("After {0} iterations theta0 = {1}, theta2 = {2},theta3={3},error = {4}".format(epochs, theta0, theta1,theta2, compute_error(theta0, theta1,theta2, x_data, y_data)))

ax=plt.figure().add_subplot(111,projection = "3d")
ax.scatter(x_data[:,0],x_data[:,1],y_data,c="r",marker="o",s=100) #点为红色三角形
x0= x_data[:,0]
x1=x_data[:,1]
#生成网格矩阵
x0,x1=np.meshgrid(x0,x1)
z=theta0+x0*theta1+x1*theta2
#画3D图
ax.plot_surface(x0,x1,z)
#设置坐标轴
ax.set_xlabel("x0")
ax.set_ylabel("x1")
ax.set_zlabel("y")
#显示
plt.show()
                

                
                
                
#调用scikit learn库实现多元线性回归                
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
# 载入数据
data=np.array(pd.read_excel("财政收入.xls"))
x_data=data[:,1:]
y_data=data[:,0]
#plt.scatter(x_data,y_data)
#plt.show()
#print(x_data.shape) 
#/*输出结果为(100,)，是一维数组的意思，但是sklearn需要接受二维数组，
#	所以要将一维数组转成二维数组，即下面两行代码
#*/
#x_data = data[:,0,np.newaxis]
#y_data = data[:,1,np.newaxis]
# 创建并拟合模型
model = LinearRegression()
model.fit(x_data, y_data)
model.coef_
model.intercept_
y_pre=model.predict(x_data)
# 画图
ax=plt.figure().add_subplot(111,projection = "3d")
ax.scatter(x_data,y_data,c="r",marker="o",s=100) #点为红色三角形
x0= x_data[:,0]
x1=x_data[:,1]
#生成网格矩阵
x0,x1=np.meshgrid(x0,x1)
z=model.intercept_+x0*model.coef_[0]+x1*model.coef_[1]
#画3D图
ax.plot_surface(x0,x1,z)
#设置坐标轴
ax.set_xlabel("x0")
ax.set_ylabel("x1")
ax.set_zlabel("y")
#显示
plt.show()
                
                
                
                
                
                

import seaborn as sns
sns.set(style="ticks")

# Load the example dataset for Anscombe's quartet

df=pd.read_excel("财政收入.xls")
# Show the results of a linear regression within each dataset
sns.lmplot(x="x1", y="y", data=df,aspect=1.5
           , order=2,ci=0.95, palette="Set3") 
           
                
                
                
                