# !/usr/bin/python3
# -*- codeing = utf-8 -*-
# @Time : 7/29/2021 5:26 PM
# @Author : Liu
# @File : test_stack.py
# @Software : PyCharm
from sklearn import svm as SVM
from sklearn.ensemble import RandomForestClassifier as RF
import xgboost as XGB
from stacking import Stacking

param_grid = {
    'rf': {
        'max_depth': [6],
        'n_estimators': [1800],
    },
    'xgb': {
        'max_depth': [10,],
        'learning_rate': [0.005],
        'n_estimators': [1200],
    },
    'svm': {
        "kernel": ['rbf'],
        "gamma": [ 0.001953125,],
        "C": [32,],
        'probability': [True],
    },

}
SVM = SVM.SVC


x = [[1,2,3,4,5],[1,2,3,4,5],[1,1,1,1,1],[1,1,2,1,1],[1,2,3,4,5],[1,2,3,4,5],[1,1,1,1,2],[1,1,2,1,1]]
y = [1,1,0,0,1,1,0,0]
x_test = [[1,2,3,4,5],[1,2,3,4,5],[1,1,1,1,1]]
y_test = [1,1,0]

a = Stacking(3,'AUROC','score') # a = stacking(层数，评价指标，用class还是得分score进行下一步的训练)
a.xlf_append(1,SVM,param_grid['svm'])
a.xlf_append(1,XGB,param_grid['xgb'])

a.xlf_append(2,SVM,param_grid['svm'])
a.xlf_append(2,RF,param_grid['rf'])

a.xlf_append(3,SVM,param_grid['svm'])
a.layer_connect_layer(1,2)
a.layer_connect_layer(2,3)
a.layer_connect_layer(1,3)

a.set_data(x,y)

a.train_stacking()

cc = a.predict(x_test)
dd = a.predict_proba(x_test)
# 或者
cc = a.predict(x_test,y_test)
dd = a.predict_proba(x_test,y_test)
print(cc)
print(dd)

