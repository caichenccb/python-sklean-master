# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 14:57:55 2019

@author: 老师
"""

###一元线性回归###

#statsmodels运行出现scipy库相关错误注意！！！
#statsmodels只能连接scipy1.2.0版本
#打开othersetting的库安装目录script，输入cmd打开命令输入窗口，输入以下代码即可
#pip install scipy==1.2.0 -i https://pypi.doubanio.com/simple/  --trusted-host pypi.doubanio.com  django
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
data = pd.read_csv('财政收入.csv',encoding="gbk")
y = data['y']
x1 = data['x1']
x = sm.add_constant(x1)#添加截距项#,回归方程添加一列x0=1
model = sm.OLS(y, x)
#数据拟合，生成模型
results = model.fit()
print(results.summary())
#应用模型预测作图
#用原始的x绘图会出错，因为代码sm.add_constant(x1)多加一列1,用x['x1']或者x1绘图即可
Y=results.fittedvalues  #预测值，原先的数据，用回归函数得到的估计值
plt.plot(x['x1'], y, 'b.', label='data')#原始数据
plt.plot(x['x1'], Y, 'r-',label='test')#拟合数据
#plt.legend(loc='best') #展示各点表示意思，即label
plt.show()


#####多元回归####

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
data = pd.read_csv('财政收入.csv')
y = data['y']
x1 = data['x1']
x2 = data['x2']
x3 = data['x3']
x = np.column_stack((x1, x2,x3))#表示自变量包含x1和x1的平方，如果需要更高次，可再做添加，也可以添加交叉项x1*y，或者其他变量
#注意自变量设置时，每项高次都需依次设置,x1**2表示乘方，或者x1*x1也可以
x = sm.add_constant(x) #增一列1，用于分析截距
model = sm.OLS(y, x)
#数据拟合，生成模型
result = model.fit()
print(result.summary())
#应用模型预测
Y=result.fittedvalues  #预测值，原先的数据，用回归函数得到的估计值
#######注意啦######
#也可以根据y和不同x之间实际的关系图情况，进行多元多次绘制
#如x = np.column_stack((x1, x2,x3,x1**2,x2**2,x3**2,x1*x2))


#####一元高次回归（可包含高次项）####

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
data = pd.read_csv('财政收入.csv')
y = data['y']
x1 = data['x1']
x = np.column_stack((x1, x1**2,x1*x1*x1))#表示自变量包含x1和x1的平方，如果需要更高次，可再做添加，也可以添加交叉项x1*y，或者其他变量
#注意自变量设置时，每项高次都需依次设置,x1**2表示乘方，或者x1*x1也可以
x = sm.add_constant(x) #增一列1，用于分析截距
model = sm.OLS(y, x)
#数据拟合，生成模型
result = model.fit()
print(result.summary())
#应用模型预测作图
#用原始的x绘图会出错，因为代码sm.add_constant(x1)多加一列1,用x['x1']或者x1绘图即可
Y=result.fittedvalues  #预测值，原先的数据，用回归函数得到的估计值
plt.plot(x1, y, 'b.', label='data')#原始数据
plt.plot(x1, Y, 'r-',label='test')#拟合数据
#plt.legend(loc='best') #展示各点表示意思，即label
plt.show()

#相关分析

# 定义函数corrcoef_loop,要求必须导入numpy、scipy的库，且数组array形式导入
#python索引中第一个索引号都是0
def corrcoef_loop(matrix):#必须是数组array格式，不能是dataframe格式（直接导入格式）
    rows, cols = matrix.shape[0], matrix.shape[1]#shape表示数组的size，0是行，1是列
    r = np.ones(shape=(cols, cols))
    p = np.ones(shape=(cols, cols))
    for i in range(cols):#遍历方式索引，默认step＝1，start＝0,迭代到rows的数据
        for j in range(i+1, cols):#遍历方式索引，默认step＝1，start＝i+1,迭代到rows的数据
            r_, p_ = stp.pearsonr(matrix[i], matrix[j])#pearsonr(x,y)表示生成两列数相关性检验的相关系数r和p值
            r[i, j] = r[j, i] = r_
            p[i, j] = p[j, i] = p_
    return r, p




# 调用函数
import numpy as np  # 库的载入
import pandas as pd  # 库的载入
import scipy.stats as stp  # 统计检验库
data1 = pd.read_csv('财政收入.csv')
data2=np.array(data1)
[r,p]=corrcoef_loop(data2)
print('r=')
print('\n')
print(r)
print('\n')
print('p=')
print('\n')
print(p)

#数据透视功能
import numpy as np
import pandas as pd
import datetime as dt #系统自带
data=pd.read_csv('data3.csv')
out=pd.pivot_table(data, index='kh', values='je',aggfunc=[np.average,np.sum,len])
#可以多字段index=['a','b'],返回values的均值，aggfunc增加返回
print(out)
