c# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:40:13 2019

@author: 92156
"""

#https://zhuanlan.zhihu.com/p/22692029
#文章来源 

#statsmodels.OLS 的输入有 (endog, exog, missing, hasconst) 四个
#我们现在只考虑前两个。
#第一个输入 endog 是回归中的反应变量（也称因变量），
#是上面模型中的 y(t), 输入是一个长度为 k 的 array。
#第二个输入 exog 则是回归变量（也称自变量）的值
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

#注意，statsmodels.OLS 不会假设回归模型有常数项
Y=

#没有专门的数值全为1的一列，
#Statmodels 有直接解决这个问题的函数：
#sm.add_constant()。
#它会在一个 array 左侧加上一列 1。
#（本文中所有输入 array 的情况也可以使用同等的 list、pd.Series 或 pd.DataFrame。）


print ("OLS统计结果解释")

#可决系数   R-squared
#R-squared误差的大小意味着模型的拟合度的好坏。   Adj. R-squared:  
#F统计量 F-statistic: 
#F-statistic是F分布下的统计量 
#对数似然函数的值   Log-Likelihood:  越小越好
#AIC和BIC是似然函数和乘法函数的线性组合，不同之处在于惩罚力度不同。越小越好。
#coeff指的是回归系数，std.err指的是估计标准误，T的数值表示的是对回归参数的显著性检验值。p值就是显著性，95%CI就是回归系数95%的置信区间。
#Jarque-Bera的P值越近于0，表明日收益率数据服从正态分布。
#Durbin-Watson:判断残差序列是否存在一阶自相关，     Durbin-Waston值是2.112说明不存在自相关
#Omnibus:    模型系数的混合检验
