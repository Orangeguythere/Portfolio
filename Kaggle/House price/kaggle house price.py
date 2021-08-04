import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.api as sp
import pylab as pl
import seaborn as sns
from numpy import arange
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from sklearn import preprocessing
from sklearn import utils
from sklearn.metrics import roc_auc_score
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.tree import export_graphviz
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import RepeatedKFold
import scipy.stats as stats
import xgboost as xgb

#Remove sklearn warning
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


# Performance metrics
from sklearn.metrics import accuracy_score, classification_report
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process




#Data
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

#Improve train data
for column in train:
    if train[column].dtypes=="object":
        #Change NA value to 0 for categorical variables
        train[column]=train[column].fillna(0)
        #Change object value to category
        train[column] = pd.Categorical(train[column])
        #Transform str category to number 
        train[column] = train[column].cat.codes
    else:
        #Fill NA with average
        train[column]=train[column].fillna(train[column].mean())
    
print(train.info())

#Improve test data
for column in test:
    if test[column].dtypes=="object":
        #Change NA value to 0 for categorical variables
        test[column] = test[column].fillna(0)
        #Change object value to category
        test[column] = pd.Categorical(test[column])
        #Transform str category to number 
        test[column] = test[column].cat.codes
    else:
        #Fill NA with average
        test[column]=test[column].fillna(test[column].mean())
    
print(test.info())


#First filter analysis : correlation matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(2, 15))
sns.heatmap(corrmat[["SalePrice"]], square=True,vmin=-1, vmax=1,cmap='BrBG', annot=True)
dfcorrmat=pd.DataFrame(corrmat)

#dftrain show variables based on correlation level
dftrain=pd.DataFrame()
A=0
#N as the correlation level
N=0.3
while A< len(dfcorrmat["SalePrice"][dfcorrmat["SalePrice"]>N]):
    dftrain=dftrain.append(dfcorrmat[dfcorrmat["SalePrice"][dfcorrmat["SalePrice"]>N].index[A]])
    #print(dfcorrmat["SalePrice"][dfcorrmat["SalePrice"]>N].index[A])
    #print(dfcorrmat["SalePrice"][dfcorrmat["SalePrice"]>N][A])
    A+=1
A=0
while A< len(dfcorrmat["SalePrice"][dfcorrmat["SalePrice"]<-N]):
    dftrain=dftrain.append(dfcorrmat[dfcorrmat["SalePrice"][dfcorrmat["SalePrice"]<-N].index[A]])
    #print(dfcorrmat["SalePrice"][dfcorrmat["SalePrice"]<-N].index[A])
    #print(dfcorrmat["SalePrice"][dfcorrmat["SalePrice"]<-N][A])
    A+=1

#df1 = df[['a','d']]
dftrain = train[dftrain["SalePrice"].index.values.tolist()]
X_train = dftrain.drop(["SalePrice"],1) 
dftest = test[X_train.columns]
print(dftest)
#plt.show()


#Real test
#X_train = train.drop(["Id","SalePrice"],1) 
X_train = dftrain.drop(["SalePrice"],1) 
X_test = dftest
y_train = dftrain["SalePrice"]
#print(X_train,X_test,y_train)
#y_test=test["SalePrice"]
"""
#Train test split for scoring purpose
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3)
print(X_test,y_test)
"""

#Regression problems for predicting a quantity, using some regression algorithms

#Regression linear simple
# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(X_train, y_train)
# Make predictions using the testing set
y_predRL = regr.predict(X_test)
print(y_predRL)


#Random forest regressor
clf=RandomForestRegressor(n_estimators=100)
#Train the model using the training sets 
clf.fit(X_train, y_train)
y_predRF=clf.predict(X_test)
print(y_predRF)



#ElasticNet Regression model
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define model
ratios = arange(0, 1, 0.1)
alphas = [ 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
model = ElasticNetCV(l1_ratio=ratios, alphas=alphas, cv=cv, n_jobs=-1)
# fit model
model.fit(X_train, y_train)
# make a prediction
yhat = model.predict(X_test)
#print(yhat)

#Adaboost 
model = AdaBoostRegressor()
# evaluate the model
model.fit(X_train, y_train)
yada = model.predict(X_test)

#xgBoost
model=xgb.XGBRegressor(
learning_rate =0.1,
n_estimators=100,
reg_lambda=1,
gamma=0,
max_depth=3)
    
#xgb_model = xgboost.XGBRegressor(learning_rate =0.1, n_estimators=1000, max_depth=5,
#min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, nthread=6, scale_pos_weight=1, seed=27)

model.fit(X_train, y_train)
yxgb = model.predict(X_test)


importance=clf.feature_importances_
feature_imp = pd.Series(importance,index=X_train.columns).sort_values(ascending=False)
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Score')
plt.ylabel('Features')
plt.title("Important features")
plt.legend()
#plt.show()
  
#Avergae of all prediction
avg=(y_predRL+y_predRF+yada+yxgb)/4
"""
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_predRL))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_predRL))
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_predRF))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_predRF))
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, yada))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, yada))
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, yxgb))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, yxgb))
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, avg))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, avg))
"""    


#Final submission
sample = pd.read_csv('sample_submission.csv')
sample["SalePrice"]=avg
print(sample)
sample.to_csv('Finalsub.csv', index = False)

#First try score (RMSLE) 2021 = 0.154
#(Whitout data enginering)




















#Old version 2019
"""
profile = ProfileReport(train, title='Pandas Profiling Report', html={'style':{'full_width':True}})
print(profile)
profile.to_file(output_file="your_report.html")

#Reshape
train["LotFrontage"]= train["LotFrontage"].fillna(train["LotFrontage"].mean())
test["LotFrontage"]= test["LotFrontage"].fillna(test["LotFrontage"].mean())

train["Alley"]= train["Alley"].fillna("No Alley", inplace = False)
test["Alley"]= test["Alley"].fillna("No Alley", inplace = False)

train["MasVnrType"]= train["MasVnrType"].fillna("None", inplace = False)
test["MasVnrType"]= test["MasVnrType"].fillna("None", inplace = False)

train["MasVnrArea"]= train["MasVnrArea"].fillna(0, inplace = False)
test["MasVnrArea"]= test["MasVnrArea"].fillna(0, inplace = False)

train["BsmtQual"]= train["BsmtQual"].fillna("None", inplace = False)
test["BsmtQual"]= test["BsmtQual"].fillna("None", inplace = False)
train["BsmtCond"]= train["BsmtCond"].fillna("None", inplace = False)
test["BsmtCond"]= test["BsmtCond"].fillna("None", inplace = False)
train["BsmtExposure"]= train["BsmtExposure"].fillna("None", inplace = False)
test["BsmtExposure"]= test["BsmtExposure"].fillna("None", inplace = False)
train["BsmtFinType1"]= train["BsmtFinType1"].fillna("None", inplace = False)
test["BsmtFinType1"]= test["BsmtFinType1"].fillna("None", inplace = False)
train["BsmtFinType2"]= train["BsmtFinType2"].fillna("None", inplace = False)
test["BsmtFinType2"]= test["BsmtFinType2"].fillna("None", inplace = False)

train["Electrical"]= train["Electrical"].fillna("None", inplace = False)
test["Electrical"]= test["Electrical"].fillna("None", inplace = False)

train["FireplaceQu"]= train["FireplaceQu"].fillna("None", inplace = False)
test["FireplaceQu"]= test["FireplaceQu"].fillna("None", inplace = False)

train["GarageType"]= train["GarageType"].fillna("None", inplace = False)
test["GarageType"]= test["GarageType"].fillna("None", inplace = False)
train["GarageYrBlt"]= train["GarageYrBlt"].fillna("None", inplace = False)
test["GarageYrBlt"]= test["GarageYrBlt"].fillna("None", inplace = False)
train["GarageFinish"]= train["GarageFinish"].fillna("None", inplace = False)
test["GarageFinish"]= test["GarageFinish"].fillna("None", inplace = False)
train["GarageQual"]= train["GarageQual"].fillna("None", inplace = False)
test["GarageQual"]= test["GarageQual"].fillna("None", inplace = False)
train["GarageCond"]= train["GarageCond"].fillna("None", inplace = False)
test["GarageCond"]= test["GarageCond"].fillna("None", inplace = False)

train["PoolQC"]= train["PoolQC"].fillna("None", inplace = False)
test["PoolQC"]= test["PoolQC"].fillna("None", inplace = False)

train["Fence"]= train["Fence"].fillna("None", inplace = False)
test["Fence"]= test["Fence"].fillna("None", inplace = False)

train["MiscFeature"]= train["MiscFeature"].fillna("None", inplace = False)
test["MiscFeature"]= test["MiscFeature"].fillna("None", inplace = False)

#Reshape plus precis pour valeur test manquantes
test["SaleType"]= test["SaleType"].fillna("Oth", inplace = False)
test["GarageArea"]= test["GarageArea"].fillna(0, inplace = False)
test["GarageCars"]= test["GarageCars"].fillna(0, inplace = False)
test["Functional"]= test["Functional"].fillna("Typ", inplace = False)
test["KitchenQual"]= test["KitchenQual"].fillna("TA", inplace = False)
test["BsmtHalfBath"]= test["BsmtHalfBath"].fillna(0, inplace = False)
test["BsmtFullBath"]= test["BsmtFullBath"].fillna(0, inplace = False)
test["TotalBsmtSF"]= test["TotalBsmtSF"].fillna(0, inplace = False)
test["BsmtUnfSF"]= test["BsmtUnfSF"].fillna(0, inplace = False)
test["BsmtFinSF2"]= test["BsmtFinSF2"].fillna(0, inplace = False)
test["BsmtFinSF1"]= test["BsmtFinSF1"].fillna(0, inplace = False)
test["Exterior2nd"]= test["Exterior2nd"].fillna("CBlock", inplace = False)
test["Exterior1st"]= test["Exterior1st"].fillna("CBlock", inplace = False)
test["Utilities"]= test["Utilities"].fillna("NoSewr", inplace = False)
test["MSZoning"]= test["MSZoning"].fillna("RH", inplace = False)
"""
"""
#correlation matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()
"""
#We keep : OverallQual,GrLivArea,GarageCars,TotalBsmtSF,FullBath

"""
#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size = 2.5)
plt.show()
"""

"""
#Verification valeur nulle
item=0
while item < len(train.isnull().sum()):
    print(str(train.columns.values[item]) + " : "+str(train.isnull().sum()[item]))
    
    item+=1

item=0
while item < len(test.isnull().sum()):
    print(str(test.columns.values[item]) + " : "+str(test.isnull().sum()[item]))
    
    item+=1
"""
"""
#Schema sous forme de pie chart 
df = train["Alley"].value_counts()
df.plot(kind='pie', subplots=True, figsize=(8, 8),autopct='%1.1f%%')
plt.title("Test")
plt.ylabel("")
plt.show()
"""
"""
#Reshape string into float
le = preprocessing.LabelEncoder()
n=0
while n < len(train.dtypes):
    if train.dtypes[n] not in ("int64","float64"):
        print(train.dtypes[n])


        row=0
        liste=[]
        while row < len(train):
            if train.iloc[row, n]:
                liste.append(train.iloc[row, n])
                #value=le.fit(train.iloc[row, n])
                #train.at[row,train.columns.values[n]]= le.transform(value)
            row+=1

        le.fit(liste)
        #print(train.columns.values[n])
        #print(list(le.classes_))
        #print(le.transform(list(le.classes_)))
        train[train.columns.values[n]]=le.fit_transform(liste)
        #print (train[train.columns.values[n]])

    else:
        1
    n+=1

n=0
while n < len(test.dtypes):
    if test.dtypes[n] not in ("int64","float64"):
        print(test.dtypes[n])


        row=0
        liste=[]
        while row < len(test):
            if test.iloc[row, n]:
                liste.append(test.iloc[row, n])
                #value=le.fit(test.iloc[row, n])
                #test.at[row,test.columns.values[n]]= le.transform(value)
            row+=1

        le.fit(liste)
        #print(test.columns.values[n])
        #print(list(le.classes_))
        #print(le.transform(list(le.classes_)))
        test[test.columns.values[n]]=le.fit_transform(liste)
        #print (test[train.columns.values[n]])
    

        #df.at[0,’Age']= 20

    else:
        1
    n+=1



#X=train.drop("SalePrice", axis=1)
X=train[["OverallQual","GrLivArea","GarageCars","TotalBsmtSF","FullBath"]]
y=train["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


Methodes=[
#Ensemble Methods
ensemble.AdaBoostClassifier(),
ensemble.BaggingClassifier(),
ensemble.ExtraTreesClassifier(),
#ensemble.GradientBoostingClassifier(),
ensemble.RandomForestClassifier(),

#Gaussian Processes
#gaussian_process.GaussianProcessClassifier(),

#GLM
#linear_model.LogisticRegressionCV(),
linear_model.LogisticRegression(C=1000,random_state=0, solver='liblinear'),
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


#Vrai test
ok=discriminant_analysis.LinearDiscriminantAnalysis()
ok.fit(X,y)
y_pred = ok.predict(test[["OverallQual","GrLivArea","GarageCars","TotalBsmtSF","FullBath"]])
print(y_pred)


#Send file
new=[]
for item in test["Id"]:
    new.append([item,y_pred[item-1461]])


final=pd.DataFrame(new, columns = ['Id', 'SalePrice'])
print(final)
final.to_csv('Finalsub.csv', index = False)
"""