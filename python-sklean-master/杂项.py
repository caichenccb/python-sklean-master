py# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 11:22:39 2019

@author: 92156
"""
reset_selective bool # 删除某一变量
reset #重置所有变量
#首先先看数据总体
cc1.info()
cc1.describe()
绘图的时候要设置字体
my_font = font_manager.FontProperties(fname="C:\Windows\Fonts\msyh.ttc")
注意title()、xlabel()、ylabel()中设置字体的参数为fontproperties，legend()设置字体参数为prop
plt.title("回家近几年",fontproperties=my_font) 
或者
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


合并dataframe
res = pd.concat([df,qq],axis=1) 

统计分析

def status(x):
    import pandas as pd
 return pd.Series([x.count(),x.min(),x.idxmin(),x.quantile(.25),x.median(),
                      x.quantile(.75),x.mean(),x.max(),x.idxmax(),x.mad(),x.var(),
                      x.std(),x.skew(),x.kurt()],index=['总数','最小值','最小值位置','25%分位数',
                    '中位数','75%分位数','均值','最大值','最大值位数','平均绝对偏差','方差','标准差','偏度','峰度'])
 
计数累加
ss=cc10["qhx:je"]
a=0
b=0
c=0
d=0
f=0
e=0
g=0
k=0
h=0
j=0
for x in ss:
    if x<500 :
      
        a=a+1
    elif x<1000:
    
        b=b+1
    elif x<2000:
       
        c=c+1
    elif x<3000:    
     
        d=d+1
    elif x<6000:
        
        f=f+1
    elif x<10000:
     
        e=e+1
    elif x<15000:
   
        g=g+1
    elif x<25000:
      
        h=h+1
    elif x<35000:
      
        j=j+1
    elif x >35001:
 
        k=k+1
        
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

#玫瑰作图
from windrose import WindroseAxes
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from numpy.random import random
from numpy import arange
ws = random(500)*6
wd = random(500)*360
#A quick way to create new windrose axes...
def new_axes():
    fig = plt.figure(figsize=(8, 8), dpi=80, facecolor='w', edgecolor='w')
    rect = [0.1, 0.1, 0.8, 0.8]
    ax = WindroseAxes(fig, rect, axisbg='w')
    fig.add_axes(ax)
    return ax
#...and adjust the legend box
def set_legend(ax):
    l = ax.legend(shadow=True, bbox_to_anchor=[-0.1, 0], loc='lower left')
    plt.setp(l.get_texts(), fontsize=10)
  
ax = new_axes()
ax.bar(wd, ws, normed=True, opening=0.8, edgecolor='white')
set_legend(ax)
plt.show()

#灰度图
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns

points=np.arange(-5,5,0.01)
xs,ys=np.meshgrid(points,points)
ys
z=np.sqrt(ys**2+xs**2)
z
plt.imshow(z,cmap=plt.cm.tab20b);
#cm.的色条
#https://matplotlib.org/examples/color/colormaps_reference.html
#https://matplotlib.org/users/colormaps.html
plt.colorbar()
plt.title("xxxxx")

#删空值
aa=cc1.isnull()
bb=cc1.dropna()
#随机抽样
df=pd.DataFrame(np.arange(5*4).reshape((5,4)))
sampler=np.random.permutation(5)
df.take(sampler)
#或者 
df.sample(n=3)  # 随机抽取3个数量

#清除空格和拆分
piced=[x.strip() for x in val.split(",")]

#merge函数的运用
cc7=pd.merge(cc2,cc6,on=["spbm","dtime","je"],suffixes=("_left","_right"))   #本店会员


#老师的    利用pyecharts 来做
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pygal as pyg#绘图库
import pyecharts as pyec  #血的教训，网上的教程都是针对0.5版本，1.0之后的版本完全不同

#其他题目按照要求替换数据
data=pd.read_excel('作业1.xlsx',encoding='gbk')   #其他题目，更改此处数据
#利用python的分组操作工具，进行分组和数据统计
#数据操作也必须用Python进行，excel处理不了大量数据
group1=data['社会支持总分'].groupby(data['性别']) #对社会支持总分按照性别分组，注意输出的是结构而不是数据表，只能进行求和等统计操作，不能展示
print(group1.count())
#通过这里显示的数据，修改到下面的饼图数据中即可

index =['男','女']
value=[1882,1635]  #该数据根据上面group求和的输出结果
pie =pyec.Pie('')#可书写title
pie.add('',index,value,is_label_show=True)
pie.render('性别比.html')#文件夹中出现的性别比.html打开就是交互图片

pie1 =pyec.Pie('')
pie1.add('',index,value,radius=[40,75],is_label_show = True)#radius表示圆环内外圈半径
pie1.render('性别比_环.html')

pie2=pyec.Pie('')
pie2.add( '',index,value,radius=[65, 75],center=[50, 50],is_label_show=True)
pie2.add('',index,value,radius=[0, 60],center=[50, 50],rosetype='area')
pie2.render('性别比_玫瑰.html')

group2=data['社会支持总分'].groupby(data['年级号']) #对社会支持总分按照性别分组，注意输出的是结构而不是数据表，只能进行求和等统计操作，不能展示
print(group2.count())
pie3 =pyec.Pie('')
pie3.add('',['一年级','二年级','三年级','四年级','五年级'],[1294,1215,548,162,298],is_label_show=True)
pie3.render('年级比.html')
pie4=pyec.Pie('')
pie4.add( '',['一年级','二年级','三年级','四年级','五年级'],[1294,1215,548,162,298],radius=[65, 75],center=[50, 50],is_label_show=True,legend_pos='right')
pie4.add('',['一年级','二年级','三年级','四年级','五年级'],[1294,1215,548,162,298],radius=[0, 60],center=[50, 50],rosetype='area')
pie4.render('年级比_玫瑰.html')


凡小于百分之1分位数和大于百分之99分位数的值将会被百分之1分位数和百分之99分位数替代：
def cc(x,quantile=[0.01,0.99]):
#    """盖帽法处理异常值
#    Args：
#        x：pd.Series列，连续变量
#        quantile：指定盖帽法的上下分位数范围
#    """

# 生成分位数
    Q01,Q99=x.quantile(quantile).values.tolist()