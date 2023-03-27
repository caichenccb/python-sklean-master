# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 13:50:50 2019

@author: 92156
"""


#https://zhuanlan.zhihu.com/p/43836485
import sklearn
from sklearn.datasets import load_boston
import numpy as np
dataset=load_boston()
type(dataset)#数据类型是sklearn的数据集类型
sklearn.datasets.base.Bunch
dataset.data.shape#自变量的维度
#(506, 13)
dataset.target.shape#因变量的维度
#(506,)
from sklearn.cross_validation import train_test_split
# 随机采样25%的数据构建测试样本，其余作为训练样本。
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target,random_state=33, test_size=0.25)
# 分析回归目标值的差异。
print ("The max target value is", np.max(dataset.target))
print ("The min target value is", np.min(dataset.target))
print ("The average target value is", np.mean(dataset.target))
#The max target value is 50.0
#The min target value is 5.0
#The average target value is 22.5328063241
import matplotlib.pyplot as plt
plt.hist(dataset.target)
plt.show()
plt.hist(dataset.data)
plt.show()

#标准化数据
from sklearn.preprocessing import StandardScaler
ss_x=StandardScaler()
ss_y=StandardScaler()
X_train=ss_x.fit_transform(X_train)
X_test=ss_x.transform(X_test)
y_train=ss_y.fit_transform(y_train)
y_test=ss_y.transform(y_test)

#基于最小二乘法的LinearRegression：


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)#拟合模型
lr_y_predict = lr.predict(X_test)#做预测


#评估模型
# 使用LinearRegression模型自带的评估模块，并输出评估结果。
print ('The value of default measurement of LinearRegression is', lr.score(X_test, y_test))
#The value of default measurement of LinearRegression is 0.6763403831
# 从sklearn.metrics依次导入r2_score、mean_squared_error以及mean_absoluate_error用于回归性能的评估。
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# 使用r2_score模块，并输出评估结果。
print ('The value of R-squared of LinearRegression is', r2_score(y_test, lr_y_predict))
#The value of R-squared of LinearRegression is 0.6763403831
#可视化因变量
import matplotlib.pyplot as plt
plt.scatter(np.arange(len(y_test)),y_test,color = 'red',label='y_test')
plt.scatter(np.arange(len(lr_y_predict)),lr_y_predict,color = 'blue',label='y_pred')
plt.legend(loc=2)
plt.show()



#拟合效果还不错
#
#在模型评估时，两种方式是一样的，以后直接用第一种，即模型自带的score就可以了
#
#但是，一个拟合出来的模型并不是直接可以拿来用的。还需要对其统计性质进行检验
#
#主要有以下四个检验： （数值型）自变量要与因变量有线性关系； 残差基本呈正态分布； 残差方差基本不变（同方差性）； 残差（样本）间相关独立。
#
#第一个可以直接绘制每隔变量与因变量之间的散点图（子图）,还是以波斯顿房价为例进行演示，如下
xlabel=[]
for i in range(13):
    x_i=np.array(dataset.data[:,i])
    xlabel.append(x_i)
    plt.style.use('seaborn')
    figurei=plt.subplot(2,7,i+1)
    #figurei.patch.set_facecolor('blue')
    figurei.scatter(x_i,dataset.target)
plt.show()

#检验残差是否基本上呈正态分布也建议直接Spss

from scipy import stats
stats.probplot(dataset.target,dist="norm", plot=plt)
plt.show()

#不确定，建议SPSS
d= dataset.target
sorted_ = np.sort(d)
yvals = np.arange(len(sorted_))/float(len(sorted_))
plt.plot(sorted_, yvals)
plt.show()



#这个是绘制VIF的程序，没看懂，以后再研究

import numpy as np
import matplotlib.pyplot as plt
vif2=np.zeros((15,1))
for i in range(15):
    tmp=[k for k in range(13) if k!=i]
    #clf.fit(X2[:,tmp],X2[:,i])
    vifi=1/(1-lr.score(X_test, y_test))
    vif2[i]=vifi

vif3=np.zeros((15,1))
for i in range(15):
    tmp=[k for k in range(15) if k!=i]
    #clf.fit(X3[:,tmp],X3[:,i])
    vifi=1/(1-lr.score(X_test, y_test))
    vif3[i]=vifi  

plt.figure()
ax = plt.gca()
ax.plot(vif2)
ax.plot(vif3)
plt.xlabel('feature')
plt.ylabel('VIF')
plt.title('VIF coefficients of the features')
plt.axis('tight')
plt.show()

#随机梯度下降原理拟合的线性回归模型

from sklearn.linear_model import SGDRegressor
model = SGDRegressor()
model.fit(X_train,y_train)#拟合模型
SGDRegressor(alpha=0.0001, average=False, epsilon=0.1, eta0=0.01,
       fit_intercept=True, l1_ratio=0.15, learning_rate='invscaling',
       loss='squared_loss', n_iter=5, penalty='l2', power_t=0.25,
       random_state=None, shuffle=True, verbose=0, warm_start=False)
sgdr_y_predict=model.predict(X_test)#做预测
#可视化结果y的真实值和预测值之间的差距：

plt.scatter(np.arange(len(y_test)),y_test,color = 'red',label='y_test')
plt.scatter(np.arange(len(sgdr_y_predict)),sgdr_y_predict,color = 'blue',label='y_pre')
plt.legend(loc=2)
plt.show()
