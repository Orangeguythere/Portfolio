import pandas as pd
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
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score, classification_report
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process


#Data
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
sample_submission=pd.read_csv("sample_submission.csv")

print(train)

"""
profile = ProfileReport(train, title='Pandas Profiling Report', html={'style':{'full_width':True}})
print(profile)
profile.to_file(output_file="your_report.html")
"""


liste=[]
#train=train.loc[train['sentiment'] == 'negative']
for text in train["text"]:

    try:
        for word in text.split():
            liste.append(word)
    except:
        1

print(Counter(liste))

count_vectorizer = feature_extraction.text.CountVectorizer()

train_vectors = count_vectorizer.fit_transform(train["text"].values.astype('U'))
selected_vectors = count_vectorizer.fit_transform(train["selected_text"].values.astype('U'))
print(train_vectors)

test_vectors = count_vectorizer.transform(test["text"].values.astype('U'))


#Transformation en int
col         = 'sentiment'
conditions  = [ (train[col]=="negative"), 
train[col]=="neutral", 
train[col]<="positive"]
choices     = [-1,0,1]
train["sentiment"] = np.select(conditions, choices)

print(train)


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
linear_model.LogisticRegressionCV(),
linear_model.LogisticRegression(C=100,random_state=0, solver='liblinear'),
linear_model.PassiveAggressiveClassifier(),
linear_model.RidgeClassifierCV(),
linear_model.SGDClassifier(),
linear_model.Perceptron(),

#Navies Bayes
naive_bayes.BernoulliNB(),
#naive_bayes.GaussianNB(),

#Nearest Neighbor
neighbors.KNeighborsClassifier(),

#SVM
svm.SVC(probability=True),
svm.NuSVC(probability=True),
svm.LinearSVC(),

#Trees    
tree.DecisionTreeClassifier(),
tree.ExtraTreeClassifier(),

#Discriminant Analysis
#discriminant_analysis.LinearDiscriminantAnalysis(),
#discriminant_analysis.QuadraticDiscriminantAnalysis(),
]

for methode in Methodes:
    clf = methode
    scores = model_selection.cross_val_score(clf, train_vectors, train["sentiment"], cv=3)
    print(str(methode) + str(scores))

clf=naive_bayes.BernoulliNB()
clf.fit(train_vectors, train["sentiment"])

sample_submission["target"] = clf.predict(test_vectors)
sample_submission.to_csv("final.csv", index=False)

