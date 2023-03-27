# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 20:08:10 2019

@author: 92156
"""

from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_excel("工业排污.xls",encoding = 'gbk')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def get_boxplot(data, start, end):
    fig, ax = plt.subplots(1, end-start, figsize=(24, 4))
    for i in range(start, end):
        sns.boxplot(y=data[data.columns[i]], data=data, ax=ax[i-start])
get_boxplot(df, 1,4 )
def drop_outlier(data, start, end):
    for i in range(start, end):
        field = data.columns[i]
        Q1 = np.quantile(data[field], 0.25)
        Q3 = np.quantile(data[field], 0.75)
        deta = (Q3 - Q1) * 1.5
        data = data[(data[field] >= Q1 - deta) & (data[field] <= Q3 + deta)]
    return data
del_df = drop_outlier(df, 1, 4)
print("原有样本容量:{0}, 剔除后样本容量:{1}".format(df.shape[0], del_df.shape[0]))
get_boxplot(del_df, 1, 4)

km = KMeans(n_clusters=2, random_state=10)
km.fit(del_df[["工业废气排放总量","工业废水排放总量","二氧化硫排放总量"]])
print(km.cluster_centers_) 
print(km.labels_)
del_df["label"]=km.labels_
del_df["预测"]=km.predict(del_df[["工业废气排放总量","工业废水排放总量","二氧化硫排放总量"]])
sns.barplot(x=del_df.index, y="label", data=del_df,  palette="Set3")

#看当k取何值时分类较好
import matplotlib.pyplot as plt
K = range(1, 10)
sse = []
for k in K:
    km = KMeans(n_clusters=k, random_state=10)
    km.fit(del_df[["工业废气排放总量","工业废水排放总量","二氧化硫排放总量"]])
    sse.append(km.inertia_)
plt.figure(figsize=(8, 6))
plt.plot(K, sse, '-o', alpha=0.7)
plt.xlabel("K")
plt.ylabel("SSE")
plt.show()


#当k为5时，看上去簇内离差平方和之和的变化已慢慢变小，那么，我们不妨就将球员聚为7类。如下为聚类效果的代码：

km = KMeans(n_clusters=5, random_state=10)
km.fit(del_df[["工业废气排放总量","工业废水排放总量","二氧化硫排放总量"]])
print(km.cluster_centers_) 
print(km.labels_)
del_df["label"]=km.labels_
del_df["预测"]=km.predict(del_df[["工业废气排放总量","工业废水排放总量","二氧化硫排放总量"]])
sns.barplot(x=del_df.index, y="label", data=del_df,  palette="Set3")




#层次聚类
import pandas as pd
import seaborn as sns  #用于绘制热图的工具包
from scipy.cluster import hierarchy  #用于进行层次聚类，话层次聚类图的工具包
from scipy import cluster   
import matplotlib.pyplot as plt
from sklearn import decomposition as skldec #用于主成分分析降维的包
Z = hierarchy.linkage(del_df[["工业废气排放总量","工业废水排放总量","二氧化硫排放总量"]], method ='ward',metric='euclidean')
hierarchy.dendrogram(Z,labels = del_df.index)

#利用sns 聚类
sns.clustermap(del_df[["工业废气排放总量","工业废水排放总量","二氧化硫排放总量"]],method ='ward')   # ’single 最近点算法。   ’complete   这也是最远点算法或Voor Hees算法   ’average   UPGMA算法。 ’weighted  （也称为WPGMA  ’centroid  WPGMC算法。
