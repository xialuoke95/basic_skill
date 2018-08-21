#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 22:14:39 2018

@author: zy
"""

import pandas as pd
import numpy as np
data = pd.read_csv('~/Desktop/MedicalText1.csv')
data.iloc[1,:]
drop_col = ['Unnamed: 0','name','dgc','ComplicationDesc']
need_modify_col = ['sex','HBP','DiabHistory','DiabSelf','Stroke','HeartAttack','Rabat']
modify_way = [{'女':0,'男':1},{'无':0,'有':1},{'无':0,'有':1},{'无':0,'有':1},{'无':0,'有':1},{'无':0,'有':1},{'正常': 0,'异常':1}]
y_col = 'TransferLevel'
dedup_col = 'ComplicationAmount'

train = data.drop(drop_col, axis = 1)
train = train.drop(dedup_col, axis = 1)
for col,way in zip(need_modify_col,modify_way):
    train[col] = train[col].map(way)



yGTP_modify = np.where(train['yGTP'] > 160)[0]
bloodsugar_modify = np.where(train['bloodsugar'] > 15)[0]

for row in yGTP_modify:
    train['yGTP'].iloc[row] = 160
for row in bloodsugar_modify:
    train['bloodsugar'].iloc[row] = 15

train[y_col] = (train[y_col] == 3).astype(int)
train.describe()

# 极端值：yGTP, > 160 -> 160; bloodsugar, > 15 -> 15
# 属性变量；1，2，3 astype(int)

# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model

# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model

from sklearn import metrics 
from sklearn.cross_validation import train_test_split  

train_x = train.drop(y_col,axis = 1)
train_y = train[y_col]
X_train,X_test, y_train, y_test = train_test_split(train_x,train_y,test_size=0.1, random_state=0)

model = random_forest_classifier(X_train,y_train)
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)
y_pred_prob = [tup[1] for tup in y_pred_prob]
accuracy = metrics.accuracy_score(y_test, y_pred)  #0.77
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)   
auc = metrics.roc_auc_score(y_test, y_pred_prob) # 0.6



model = random_forest_classifier(X_train,y_train)
names = list(X_train.columns)
sort_importance = sorted(zip([round(x,3) for x in model.feature_importances_], names), reverse=True)
print sort_importance

from matplotlib import pyplot as plt
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), [tup[0] for tup in sort_importance], color = 'lightblue', align = 'center')
plt.xticks(range(X_train.shape[1]), [tup[1] for tup in sort_importance], rotation = 90)
plt.xlim([-1,X_train.shape[1]])
plt.tight_layout()
plt.show()


model = decision_tree_classifier(X_train,y_train)
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)
y_pred_prob = [tup[0] for tup in y_pred_prob]
accuracy = metrics.accuracy_score(y_test, y_pred) # 0.625
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred) 
auc = metrics.roc_auc_score(y_test, y_pred_prob) # 0.62

# 得到病人的基本信息及并发症信息
def get_features(patient_str):
    pass

