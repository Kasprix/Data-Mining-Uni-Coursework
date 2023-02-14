# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 19:09:21 2021

@author: 44778
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.base import TransformerMixin
from sklearn import tree
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import train_test_split


class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

def classifyTree(X):
    
    labelEncoder = X.apply(LabelEncoder().fit_transform)
    treeData = labelEncoder.iloc[:, :-1]
    treeTarget = labelEncoder.iloc[:, -1:]
    
    X_train, X_test, y_train, y_test = train_test_split(treeData, treeTarget, test_size=0.2, random_state=0)
    
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print()
    print("Accuracy:", accuracy)
    errorRate  = 1 - accuracy
    print('Error rate:', errorRate)
    

# 1.1

df = pd.read_csv('adult.csv')
# number of instances
noOccurances = len(df)
# for question 1.4, gathering all Nan 
nullData = df[df.isnull().any(axis=1)]


# Counts no of missing values
noNans = df.isnull().sum().sum()

instanceNanCount = df.isnull().any(axis=1).sum()

# df now removes the instances with Nan values
df = df.replace('?', np.nan).dropna()
df = df.infer_objects()
data = df.iloc[:, :-1]
target = df.iloc[:, -1:]

# Calculates formulas to answer question 1
allValues = data.shape[0] * data.shape[1]
fracValues = noNans/allValues
fracInstances = instanceNanCount/noOccurances

# 1.2

labelEncoder = df.apply(LabelEncoder().fit_transform)

discreteValues = labelEncoder.apply(lambda col: col.unique())

for col in df:
    print(df[col].unique())
print()

# 1.3

treeData = labelEncoder.iloc[:, :-1]
treeTarget = labelEncoder.iloc[:, -1:]

X_train, X_test, y_train, y_test = train_test_split(treeData, treeTarget, test_size=0.2, random_state=0)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
print('Question 1.3 rates')
print("Accuracy:", accuracy)
errorRate  = 1 - accuracy
print('Error rate:', errorRate)
print('-----------')
# 1.4

# gets random instances from dataframe, post parse of nan values
randomInstance = df.sample(n = nullData.shape[0])

# join the two dataframes
frames = [nullData, randomInstance]
dPrime = pd.concat(frames)

# replace nan values with missing and split data and target
dPrime1 = dPrime.copy()
dPrime1.fillna("missing", inplace=True)

# classify function
classifyTree(dPrime1)

# replace nan values with mode of that column and split data and target
dPrime2 = dPrime.copy()
for column in dPrime2.columns:
    dPrime2[column].fillna(dPrime2[column].mode()[0], inplace=True)
  
# classify function
classifyTree(dPrime2)

