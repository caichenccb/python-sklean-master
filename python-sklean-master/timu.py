# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:16:24 2019

@author: 92156
"""

from sklearn.linear_model import Ridge,RidgeCV   # Ridge岭回归,RidgeCV带有广义交叉验证的岭回归
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
data = pd.read_excel("R日诊量.xlsx")
alphas = 10**np.linspace(- 3, 3, 100)
lr = RidgeCV(alphas=alphas,store_cv_values=True)
X=data[["Aver pres","High pres","Low pres","Aver temp","High temp","Low temp","Aver RH","Min RH"]]
y = data['Rizhenhengduanliang']
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)
lr.fit(X_train, y_train)#拟合模型
lr.score(X_test,y_test)


