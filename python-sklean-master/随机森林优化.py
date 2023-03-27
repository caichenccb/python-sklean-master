from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,roc_auc_score
import pandas as pd
cc1=pd.read_csv("rizhenduan.csv")
traindata =cc1[["Aver pres","High pres","Low pres","Aver temp","High temp","Low temp","Aver RH","Min RH"]]
test_feature = cc1["Rizhenhengduanliang"]
X_train, X_test, y_train, y_test = train_test_split(traindata,test_feature, test_size=0.23)
clf=RandomForestClassifier(n_estimators=25,max_depth=8,criterion="gini")
clf.fit(traindata,test_feature)
print(clf.estimators_)
print(clf.classes_)
print(clf.n_features_)
print(clf.feature_importances_)
