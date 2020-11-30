#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 20:19:19 2019

@author: dub
"""
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datatrain2 = pd.read_csv('telco_train_clean.csv')
datatest = pd.read_csv('telco_test_clean.csv')

# =============================================================================
# datatrain2 = pd.read_excel('myTrain.xlsx')
# datatest = pd.read_excel('myTest.xlsx')
# =============================================================================

print(datatrain2.info())

datatrain2 = datatrain.drop("FIN_STATE", axis=1)
datatrain2 = datatrain.drop("AVG_DATA_3MONTH", axis=1)
datatrain2 = datatrain.drop("AVG_DATA_3MONTH", axis=1)
datatrain2 = datatrain.drop("AVG_DATA_3MONTH", axis=1)

datatrain2["AVG_DATA_3MONTH"].min()
datatrain2["AVG_DATA_1MONTH"].median()
datatrain2["COUNT_CONNECTIONS_3MONTH"].median()
  
datatrain2["AVG_DATA_3MONTH"].plot(kind='hist', bins=10)
plt.xlabel('Latitude Value')

datatrain2["AVG_DATA_3MONTH"].fillna("527485516.0", inplace = True)
datatrain2["AVG_DATA_1MONTH"].fillna("120002591.0", inplace = True)
datatrain2["COUNT_CONNECTIONS_3MONTH"].fillna("70.0", inplace = True)

df['species'] = pd.Factor(iris.target, iris.target_names)

datatrain2['CHURN'] = pd.Categorical(datatrain2.target, iris.target_names)

datatrain2["CHURN"] = datatrain2["CHURN"].astype('category')

Y = datatrain2.iloc[:, 1]
X = datatrain2.drop("CHURN",axis=1)
x = X.values

clf = RandomForestClassifier(n_jobs=2)
clf.fit(X, Y)

datatrain2.shape

datatrain3 = pd.get_dummies(datatrain2['START_DATE'])

labels = np.array(datatrain2['CHURN'])
features= datatrain2.drop('CHURN', axis = 1)
feature_list = list(features.columns)
features = np.array(features)



