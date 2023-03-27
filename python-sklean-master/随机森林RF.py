import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
url1 = pd.read_csv("rizhenduan.csv")
# url1 = pd.DataFrame(url1)
# df = pd.read_csv(url1,header=None)

# print(url1)
url1.columns =  ['Rizhenhengduanliang',"Aver pres","Aver temp","High temp","Low temp","Aver RH","Min RH"]
# 查看几个标签 
# Class_label = np.unique(url1['Class label'])
# print(Class_label)
# 查看数据信息
# info_url = url1.info()
# print(info_url)
 
# 除去标签之外，共有13个特征，数据集的大小为178，
# 下面将数据集分为训练集和测试集
print(type(url1))
# url1 = url1.values
# x = url1[:,0]
# y = url1[:,1:]
x,y = url1.iloc[:,1:].values,url1.iloc[:,0].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
feat_labels = url1.columns[1:]
# n_estimators：森林中树的数量
# n_jobs  整数 可选（默认=1） 适合和预测并行运行的作业数，如果为-1，则将作业数设置为核心数
forest = RandomForestClassifier(n_estimators=1000, max_depth=10,random_state=0, n_jobs=-1)
forest.fit(x_train, y_train)
 
# 下面对训练好的随机森林，完成重要性评估
# feature_importances_  可以调取关于特征重要程度
importances = forest.feature_importances_
print("重要性：",importances)
x_columns = url1.columns[1:]
indices = np.argsort(importances)[::-1]
for f in range(x_train.shape[1]):
    # 对于最后需要逆序排序，我认为是做了类似决策树回溯的取值，从叶子收敛
    # 到根，根部重要程度高于叶子。
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
 
 
# 筛选变量（选择重要性比较高的变量）
threshold = 0.15
x_selected = x_train[:,importances > threshold]
 
# 可视化
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.title("红酒的数据集中各个特征的重要程度",fontsize = 18)
plt.ylabel("import level",fontsize = 15,rotation=90)
plt.rcParams['font.sans-serif'] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False
for i in range(x_columns.shape[0]):
    plt.bar(i,importances[indices[i]],color='orange',align='center')
    plt.xticks(np.arange(x_columns.shape[0]),x_columns,rotation=90,fontsize=15)
plt.show()

#另一个
    #_*_coding:utf-8_*_
import numpy as np
import pandas as pd
 
 
cc1=pd.read_csv("rizhenduan.csv")
traindata =cc1[["Aver pres","Aver temp","Aver RH"]]
test_feature = cc1["Rizhenhengduanliang"]


def random_forest_train(feature_data, test_feature):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
 
    X_train, X_test, y_train, y_test = train_test_split(feature_data,test_feature, test_size=0.23)
 
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    # 对测试集进行预测
    y_pred = model.predict(X_test)
    # 计算准确率
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    print(RMSE)
#选深度
def random_forest_parameter_tuning1(feature_data, test_feature ):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import GridSearchCV
 
    X_train, X_test, y_train, y_test = train_test_split(feature_data, test_feature, test_size=0.23)
    param_test1 = {
        'n_estimators': range(10,190, 10)
    }
    model = GridSearchCV(estimator=RandomForestRegressor(
        min_samples_split=100, min_samples_leaf=20, max_depth=8, max_features='sqrt',
        random_state=10), param_grid=param_test1, cv=5
    )
    model.fit(X_train, y_train)
    # 对测试集进行预测
    y_pred = model.predict(X_test)
    # 计算准确率
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    print(RMSE)
    return model.best_score_, model.best_params_
#另一个
#  https://blog.csdn.net/littlle_yan/article/details/82663279  
#另一个
#    https://blog.csdn.net/oh5W6HinUg43JvRhhB/article/details/94927681
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,roc_auc_score
cc1=pd.read_csv("rizhenduan.csv")
traindata =cc1[["Aver pres","High pres","Low pres","Aver temp","High temp","Low temp","Aver RH","Min RH"]]
test_feature = cc1["Rizhenhengduanliang"]
X_train, X_test, y_train, y_test = train_test_split(traindata,test_feature, test_size=0.23)
clf=RandomForestClassifier(n_estimators=60,max_depth=4,criterion="gini")
clf.fit(traindata,test_feature)
print(clf.estimators_)
print(clf.classes_)
print(clf.n_features_)
print(clf.feature_importances_)

#案例
#https://blog.csdn.net/xun527/article/details/79518289
import pandas as pd
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
 
 
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
 
#选取一些特征作为我们划分的依据
x = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']
 
# 填充缺失值
x['age'].fillna(x['age'].mean(), inplace=True)
 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
 
dt = DictVectorizer(sparse=False)
 
print(x_train.to_dict(orient="record"))
 
# 按行，样本名字为键，列名也为键，[{"1":1,"2":2,"3":3}]
x_train = dt.fit_transform(x_train.to_dict(orient="record"))
 
x_test = dt.fit_transform(x_test.to_dict(orient="record"))
 
# 使用决策树
dtc = DecisionTreeClassifier()
 
dtc.fit(x_train, y_train)
 
dt_predict = dtc.predict(x_test)
 
print(dtc.score(x_test, y_test))
 
print(classification_report(y_test, dt_predict, target_names=["died", "survived"]))
 
# 使用随机森林
 
rfc = RandomForestClassifier()
 
rfc.fit(x_train, y_train)
 
rfc_y_predict = rfc.predict(x_test)
 
print(rfc.score(x_test, y_test))
 
