# -*- coding: utf-8 -*-
# The above encoding declaration is required and the file must be saved as UTF-8

import pandas as pd
from pandas import concat
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.api as sp
import pylab as pl
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn import utils
from sklearn.metrics import roc_auc_score
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn import feature_extraction, model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz

from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop


# Performance metrics
from sklearn.metrics import accuracy_score, classification_report
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

#import fastai
import torch
print(torch.cuda.is_available())



#Data
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
sample_submission=pd.read_csv("sample_submission.csv")

print(sample_submission)
print(train)
print(test)

#X=train.drop("SalePrice", axis=1)
X=train[["time","signal"]]
y=train["open_channels"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


Methodes=[
#Ensemble Methods
ensemble.AdaBoostClassifier(),
ensemble.BaggingClassifier(),
ensemble.ExtraTreesClassifier(),
ensemble.GradientBoostingClassifier(),
ensemble.RandomForestClassifier(),

#Gaussian Processes
#gaussian_process.GaussianProcessClassifier(),

#GLM
#linear_model.LogisticRegressionCV(),
linear_model.LogisticRegression(C=100,random_state=0, solver='liblinear'),
linear_model.PassiveAggressiveClassifier(),
linear_model.RidgeClassifierCV(),
linear_model.SGDClassifier(),
linear_model.Perceptron(),

#Navies Bayes
naive_bayes.BernoulliNB(),
naive_bayes.GaussianNB(),

#Nearest Neighbor
neighbors.KNeighborsClassifier(),

#SVM
svm.SVC(probability=True),
#svm.NuSVC(probability=True),
svm.LinearSVC(),

#Trees    
tree.DecisionTreeClassifier(),
tree.ExtraTreeClassifier(),

#Discriminant Analysis
discriminant_analysis.LinearDiscriminantAnalysis(),
#discriminant_analysis.QuadraticDiscriminantAnalysis(),
]


for methode in Methodes:
    methode.fit(X_train, y_train)
    y_model= methode.predict(X_test)
    print(str(methode) + "\n Accuracy : "+ str(accuracy_score(y_test, y_model)))


clf=ensemble.ExtraTreesClassifier()
clf.fit(X_train, y_train)

sample_submission["open_channels"] = clf.predict(test[["time","signal"]])
print(sample_submission)
sample_submission.to_csv("final.csv", float_format='%0.4f',index=False)
