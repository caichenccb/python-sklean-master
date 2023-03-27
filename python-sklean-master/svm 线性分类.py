import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


cc1=pd.read_excel("rizhenduan1.xls")

X =np.array(cc1["Aver temp"]).reshape(-1,1)
y=np.array(cc1["Rizhenhengduanliang"]).reshape(-1,1)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.23)


# contour函数是画出轮廓，需要给出X和Y的网格，以及对应的Z，它会画出Z的边界（相当于边缘检测及可视化）


polynomial_svm_clf = Pipeline([ ("poly_featutres", PolynomialFeatures(degree=3)),
                                ("scaler", StandardScaler()),
                                ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42)  )
                            ])
polynomial_svm_clf.fit( X_train, Y_train )

result =polynomial_svm_clf(X_train)  # 使用模型预测值
print('预测结果：',result)  # 输出预测值[-1. -1.  1.  1.]


clf = SVR(kernel='rbf', class_weight='balanced',)
clf.fit(X_train, Y_train)
y_predict = clf.predict(X_test)
#error = 0
#for i in range(len(X_test)):
#    if clf.predict([X_test[i]])[0] != Y_test[i]:
#        error +=1
#print( 'SVM错误率: %.4f' % (error/float(len(X_test))))
print( 'SVM精确率: ', precision_score(Y_test, y_predict, average='macro'))
print( 'SVM召回率: ', recall_score(Y_test, y_predict, average='macro'))
print( 'F1: ', f1_score(Y_test, y_predict, average='macro'))


 
from sklearn.neighbors import KNeighborsClassifier as KNN
knc = KNN(n_neighbors =6,)
knc.fit(X_train,Y_train)
y_predict = knc.predict(X_test)
print('KNN准确率',knc.score(X_test,Y_test))
print('KNN精确率',precision_score(Y_test, y_predict,  average='macro'))
print('KNN召回率',recall_score(Y_test, y_predict,  average='macro'))
print('F1',f1_score(Y_test, y_predict,  average='macro'))

 
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train)
y_predict = rfc.predict(X_test)
print('随机森林准确率',rfc.score(X_test, Y_test))
print('随机森林精确率',precision_score(Y_test, y_predict,  average='macro'))
print('随机森林召回率',recall_score(Y_test, y_predict,  average='macro'))
print('F1',f1_score(Y_test, y_predict,  average='macro'))


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.23)

# 调用模型
#svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
#svr_poly = SVR(kernel='poly', C=1e3, degree=2)
#y_rbf = svr_rbf.fit(X_train, Y_train).predict(X_train)
#y_lin = svr_lin.fit(X_train, Y_train).predict(X_train)
#y_poly = svr_poly.fit(X_train, Y_train).predict(X_train)

# 可视化结果
lw = 2
plt.scatter(X_train, Y_train, color='darkorange', label='data')
#plt.plot(X_train, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(X_train, y_lin, color='c', lw=lw, label='Linear model')
#plt.plot(X_train, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
