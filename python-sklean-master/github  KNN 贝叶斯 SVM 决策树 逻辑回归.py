# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 21:51:30 2019

@author: 92156
"""

#数据集测试
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import datasets 
data = datasets.load_iris().data     #数据不知道
X = data.iloc[:, 0:4].values
y = data.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predicted = knn.predict(X_test)
knnAccuracy = metrics.accuracy_score(y_test, predicted)
knnF1score = metrics.f1_score(y_test, predicted, average='macro')
#贝叶斯
GB = GaussianNB()
GB.fit(X_train, y_train)
predicted = GB.predict(X_test)
GBAccuracy = metrics.accuracy_score(y_test, predicted)
GBF1score = metrics.f1_score(y_test, predicted, average='macro')
#决策树
DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)
predicted = DT.predict(X_test)
DTAccuracy = metrics.accuracy_score(y_test, predicted)
DTF1score = metrics.f1_score(y_test, predicted, average='macro')
#逻辑回归
LR = LogisticRegression()
LR.fit(X_train, y_train)
predicted = LR.predict(X_test)
LRAccuracy = metrics.accuracy_score(y_test, predicted)
LRF1score = metrics.f1_score(y_test, predicted, average='macro')
#SVM
svc = SVC()
svc.fit(X_train, y_train)
predicted = svc.predict(X_test)
svcAccuracy = metrics.accuracy_score(y_test, predicted)
svcF1score = metrics.f1_score(y_test, predicted, average='macro')

result = {"accuracy": [knnAccuracy, GBAccuracy, DTAccuracy, LRAccuracy, svcAccuracy],
        "f1-score": [knnF1score, GBF1score, DTF1score, LRF1score, svcF1score]}
results = pd.DataFrame(result, index=["knn", "贝叶斯", "决策树", "逻辑回归", "SVM"])
print("iris数据集")
print(results)

#wine数据集
data = pd.read_csv(r'C:\Users\me\Desktop\wine.data.csv')
X = data.iloc[:, 1:13].values
y = data.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predicted = knn.predict(X_test)
knnAccuracy = metrics.accuracy_score(y_test, predicted)
knnF1score = metrics.f1_score(y_test, predicted, average='macro')
#贝叶斯
GB = GaussianNB()
GB.fit(X_train, y_train)
predicted = GB.predict(X_test)
GBAccuracy = metrics.accuracy_score(y_test, predicted)
GBF1score = metrics.f1_score(y_test, predicted, average='macro')
#决策树
DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)
predicted = DT.predict(X_test)
DTAccuracy = metrics.accuracy_score(y_test, predicted)
DTF1score = metrics.f1_score(y_test, predicted, average='macro')
#逻辑回归
LR = LogisticRegression()
LR.fit(X_train, y_train)
predicted = LR.predict(X_test)
LRAccuracy = metrics.accuracy_score(y_test, predicted)
LRF1score = metrics.f1_score(y_test, predicted, average='macro')
#SVM
svc = SVC()
svc.fit(X_train, y_train)
predicted = svc.predict(X_test)
svcAccuracy = metrics.accuracy_score(y_test, predicted)
svcF1score = metrics.f1_score(y_test, predicted, average='macro')

result = {"accuracy": [knnAccuracy, GBAccuracy, DTAccuracy, LRAccuracy, svcAccuracy],
        "f1-score": [knnF1score, GBF1score, DTF1score, LRF1score, svcF1score]}
results = pd.DataFrame(result, index=["knn", "贝叶斯", "决策树", "逻辑回归", "SVM"])
print("wine数据集")
print(results)

#data_banknote_authentication数据集
data = pd.read_csv(r'C:\Users\me\Desktop\data_banknote_authentication.csv')
X = data.iloc[:, 0:4].values
y = data.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predicted = knn.predict(X_test)
knnAccuracy = metrics.accuracy_score(y_test, predicted)
knnF1score = metrics.f1_score(y_test, predicted)
#贝叶斯
GB = GaussianNB()
GB.fit(X_train, y_train)
predicted = GB.predict(X_test)
GBAccuracy = metrics.accuracy_score(y_test, predicted)
GBF1score = metrics.f1_score(y_test, predicted)
#决策树
DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)
predicted = DT.predict(X_test)
DTAccuracy = metrics.accuracy_score(y_test, predicted)
DTF1score = metrics.f1_score(y_test, predicted)
#逻辑回归
LR = LogisticRegression()
LR.fit(X_train, y_train)
predicted = LR.predict(X_test)
LRAccuracy = metrics.accuracy_score(y_test, predicted)
LRF1score = metrics.f1_score(y_test, predicted)
#SVM
svc = SVC()
svc.fit(X_train, y_train)
predicted = svc.predict(X_test)
svcAccuracy = metrics.accuracy_score(y_test, predicted)
svcF1score = metrics.f1_score(y_test, predicted)

result = {"accuracy": [knnAccuracy, GBAccuracy, DTAccuracy, LRAccuracy, svcAccuracy],
        "f1-score": [knnF1score, GBF1score, DTF1score, LRF1score, svcF1score]}
results = pd.DataFrame(result, index=["knn", "贝叶斯", "决策树", "逻辑回归", "SVM"])
print("data_banknote_authentication数据集")
print(results)

#transfusion.data数据集
data = pd.read_csv(r'C:\Users\me\Desktop\transfusion.data.csv')
X = data.iloc[:, 0:4].values
y = data.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predicted = knn.predict(X_test)
knnAccuracy = metrics.accuracy_score(y_test, predicted)
knnF1score = metrics.f1_score(y_test, predicted)
#贝叶斯
GB = GaussianNB()
GB.fit(X_train, y_train)
predicted = GB.predict(X_test)
GBAccuracy = metrics.accuracy_score(y_test, predicted)
GBF1score = metrics.f1_score(y_test, predicted)
#决策树
DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)
predicted = DT.predict(X_test)
DTAccuracy = metrics.accuracy_score(y_test, predicted)
DTF1score = metrics.f1_score(y_test, predicted)
#逻辑回归
LR = LogisticRegression()
LR.fit(X_train, y_train)
predicted = LR.predict(X_test)
LRAccuracy = metrics.accuracy_score(y_test, predicted)
LRF1score = metrics.f1_score(y_test, predicted)
#SVM
svc = SVC()
svc.fit(X_train, y_train)
predicted = svc.predict(X_test)
svcAccuracy = metrics.accuracy_score(y_test, predicted)
svcF1score = metrics.f1_score(y_test, predicted)

result = {"accuracy": [knnAccuracy, GBAccuracy, DTAccuracy, LRAccuracy, svcAccuracy],
        "f1-score": [knnF1score, GBF1score, DTF1score, LRF1score, svcF1score]}
results = pd.DataFrame(result, index=["knn", "贝叶斯", "决策树", "逻辑回归", "SVM"])
print("transfusion数据集")
print(results)

#vowel-context 数据集
data = pd.read_csv(r'C:\Users\me\Desktop\vowel-context.data.csv')
X = data.iloc[:, 0:13].values
y = data.iloc[:, 13].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predicted = knn.predict(X_test)
knnAccuracy = metrics.accuracy_score(y_test, predicted)
knnF1score = metrics.f1_score(y_test, predicted, average='macro')
#贝叶斯
GB = GaussianNB()
GB.fit(X_train, y_train)
predicted = GB.predict(X_test)
GBAccuracy = metrics.accuracy_score(y_test, predicted)
GBF1score = metrics.f1_score(y_test, predicted, average='macro')
#决策树
DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)
predicted = DT.predict(X_test)
DTAccuracy = metrics.accuracy_score(y_test, predicted)
DTF1score = metrics.f1_score(y_test, predicted, average='macro')
#逻辑回归
LR = LogisticRegression()
LR.fit(X_train, y_train)
predicted = LR.predict(X_test)
LRAccuracy = metrics.accuracy_score(y_test, predicted)
LRF1score = metrics.f1_score(y_test, predicted, average='macro')
#SVM
svc = SVC()
svc.fit(X_train, y_train)
predicted = svc.predict(X_test)
svcAccuracy = metrics.accuracy_score(y_test, predicted)
svcF1score = metrics.f1_score(y_test, predicted, average='macro')

result = {"accuracy": [knnAccuracy, GBAccuracy, DTAccuracy, LRAccuracy, svcAccuracy],
        "f1-score": [knnF1score, GBF1score, DTF1score, LRF1score, svcF1score]}
results = pd.DataFrame(result, index=["knn", "贝叶斯", "决策树", "逻辑回归", "SVM"])
print("vowel-context 数据集")
print(results)

#letter 数据集
data = pd.read_csv(r'C:\Users\me\Desktop\letter.data.csv')
X = data.iloc[:, 0:16].values
y = data.iloc[:, 16].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predicted = knn.predict(X_test)
knnAccuracy = metrics.accuracy_score(y_test, predicted)
knnF1score = metrics.f1_score(y_test, predicted, average='macro')
#贝叶斯
GB = GaussianNB()
GB.fit(X_train, y_train)
predicted = GB.predict(X_test)
GBAccuracy = metrics.accuracy_score(y_test, predicted)
GBF1score = metrics.f1_score(y_test, predicted, average='macro')
#决策树
DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)
predicted = DT.predict(X_test)
DTAccuracy = metrics.accuracy_score(y_test, predicted)
DTF1score = metrics.f1_score(y_test, predicted, average='macro')
#逻辑回归
LR = LogisticRegression()
LR.fit(X_train, y_train)
predicted = LR.predict(X_test)
LRAccuracy = metrics.accuracy_score(y_test, predicted)
LRF1score = metrics.f1_score(y_test, predicted, average='macro')
#SVM
svc = SVC()
svc.fit(X_train, y_train)
predicted = svc.predict(X_test)
svcAccuracy = metrics.accuracy_score(y_test, predicted)
svcF1score = metrics.f1_score(y_test, predicted, average='macro')

result = {"accuracy": [knnAccuracy, GBAccuracy, DTAccuracy, LRAccuracy, svcAccuracy],
        "f1-score": [knnF1score, GBF1score, DTF1score, LRF1score, svcF1score]}
results = pd.DataFrame(result, index=["knn", "贝叶斯", "决策树", "逻辑回归", "SVM"])
print("letter 数据集")
print(results)

#column_2C数据集
data = pd.read_csv(r'C:\Users\me\Desktop\column_2C.data.csv')
X = data.iloc[:, 0:6].values
y = data.iloc[:, 6].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predicted = knn.predict(X_test)
knnAccuracy = metrics.accuracy_score(y_test, predicted)
#knnF1score = metrics.f1_score(y_test, predicted)
#贝叶斯
GB = GaussianNB()
GB.fit(X_train, y_train)
predicted = GB.predict(X_test)
GBAccuracy = metrics.accuracy_score(y_test, predicted)
#GBF1score = metrics.f1_score(y_test, predicted)
#决策树
DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)
predicted = DT.predict(X_test)
DTAccuracy = metrics.accuracy_score(y_test, predicted)
#DTF1score = metrics.f1_score(y_test, predicted)
#逻辑回归
LR = LogisticRegression()
LR.fit(X_train, y_train)
predicted = LR.predict(X_test)
LRAccuracy = metrics.accuracy_score(y_test, predicted)
#LRF1score = metrics.f1_score(y_test, predicted)
#SVM
svc = SVC()
svc.fit(X_train, y_train)
predicted = svc.predict(X_test)
svcAccuracy = metrics.accuracy_score(y_test, predicted)
#svcF1score = metrics.f1_score(y_test, predicted)

result = {"accuracy": [knnAccuracy, GBAccuracy, DTAccuracy, LRAccuracy, svcAccuracy]}
results = pd.DataFrame(result, index=["knn", "贝叶斯", "决策树", "逻辑回归", "SVM"])
print("column_2C数据集数据集")
print(results)

#column_3C数据集
data = pd.read_csv(r'C:\Users\me\Desktop\column_3C.data.csv')
X = data.iloc[:, 0:6].values
y = data.iloc[:, 6].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predicted = knn.predict(X_test)
knnAccuracy = metrics.accuracy_score(y_test, predicted)
#knnF1score = metrics.f1_score(y_test, predicted)
#贝叶斯
GB = GaussianNB()
GB.fit(X_train, y_train)
predicted = GB.predict(X_test)
GBAccuracy = metrics.accuracy_score(y_test, predicted)
#GBF1score = metrics.f1_score(y_test, predicted)
#决策树
DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)
predicted = DT.predict(X_test)
DTAccuracy = metrics.accuracy_score(y_test, predicted)
#DTF1score = metrics.f1_score(y_test, predicted)
#逻辑回归
LR = LogisticRegression()
LR.fit(X_train, y_train)
predicted = LR.predict(X_test)
LRAccuracy = metrics.accuracy_score(y_test, predicted)
#LRF1score = metrics.f1_score(y_test, predicted)
#SVM
svc = SVC()
svc.fit(X_train, y_train)
predicted = svc.predict(X_test)
svcAccuracy = metrics.accuracy_score(y_test, predicted)
#svcF1score = metrics.f1_score(y_test, predicted)

result = {"accuracy": [knnAccuracy, GBAccuracy, DTAccuracy, LRAccuracy, svcAccuracy]}
results = pd.DataFrame(result, index=["knn", "贝叶斯", "决策树", "逻辑回归", "SVM"])
print("column_3C数据集数据集")
print(results)