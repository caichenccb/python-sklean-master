# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 09:29:44 2019

@author: 92156
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
cc1=pd.read_excel("data5 0825.xls")
corrs=cc1[["Aver pres","Aver temp","Aver RH"]]
#使用np.corrcoef(a)可计算行与行之间的相关系数,np.corrcoef(a,rowvar=0)用于计算各列之间的相关系数,输出为相关系数矩阵。
cm=corrs.corr()
# Aver pres  Aver temp   Aver RH
#Aver pres   1.000000  -0.719097 -0.398323
#Aver temp  -0.719097   1.000000  0.549121
##Aver RH    -0.398323   0.549121  1.000000
sns.set(font_scale=1.5)   #font_scale设置字体大小
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size': 15})
# plt.tight_layout()
# plt.savefig('./figures/corr_mat.png', dpi=300)
plt.show()



cm=corrs.corr(method='spearman')
# Aver pres  Aver temp   Aver RH
#Aver pres   1.000000  -0.643182 -0.479552
#Aver temp  -0.643182   1.000000  0.487773
#Aver RH    -0.479552   0.487773  1.000000
sns.set(font_scale=1.5)   #font_scale设置字体大小
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size': 15})
# plt.tight_layout()
# plt.savefig('./figures/corr_mat.png', dpi=300)
plt.show()

cm=corrs.corr('kendall')      # Kendall Tau相关系数
# Aver pres  Aver temp   Aver RH
#Aver pres   1.000000  -0.719097 -0.398323
#Aver temp  -0.719097   1.000000  0.549121
##Aver RH    -0.398323   0.549121  1.000000
sns.set(font_scale=1.5)   #font_scale设置字体大小
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size': 15})
# plt.tight_layout()
# plt.savefig('./figures/corr_mat.png', dpi=300)
plt.show()


#Aver pres  Aver temp   Aver RH
#Aver pres   1.000000  -0.472777 -0.375974
#Aver temp  -0.472777   1.000000  0.356438
#Aver RH    -0.375974   0.356438  1.000000