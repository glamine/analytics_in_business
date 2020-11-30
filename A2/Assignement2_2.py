#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 15:52:00 2019

@author: dub
"""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import pydot

datatest = pd.read_csv('telco_test_clean.csv') 
datatest2 = datatest.drop("FIN_STATE", axis=1)



data = pd.read_csv('telco_train_clean.csv')
data2 = data.drop("FIN_STATE", axis=1)
data2 = data2.drop("START_DATE", axis=1)
data2['CHURN'] = pd.Categorical("CHURN")
data2["CHURN"] = data2["CHURN"].astype('category')
data2["PREPAID"] = data2["PREPAID"].astype('category')

data2.info()


median_value=data2['COUNT_CONNECTIONS_3MONTH'].median()
data2['COUNT_CONNECTIONS_3MONTH']=data2['COUNT_CONNECTIONS_3MONTH'].fillna(median_value)
median_value=data2['AVG_DATA_1MONTH'].median()
data2['AVG_DATA_1MONTH']=data2['AVG_DATA_1MONTH'].fillna(median_value)
median_value=data2['AVG_DATA_3MONTH'].median()
data2['AVG_DATA_3MONTH']=data2['AVG_DATA_3MONTH'].fillna(median_value)

labels = np.array(data2['CHURN'])
features= data2.drop('CHURN', axis = 1)
feature_list = list(features.columns)
features = np.array(features)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


rf = RandomForestClassifier(n_estimators = 10000, random_state = 42)
rf.fit(train_features, train_labels);

predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)
sum(errors)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

tree = rf.estimators_[5]

export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)

(graph, ) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')

mape = np.mean(100 * (sum(errors) / sum(test_labels)))
158/1250

clf = RandomForestClassifier(n_jobs=2)

clf.fit(train_features, train_labels)

preds = clf.predict(test_features)

err = abs(preds - test_labels)


sum(err)
