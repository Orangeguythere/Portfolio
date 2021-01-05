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
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, classification_report
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop



#Data
train=pd.read_csv("sales_train.csv")
#train=train.dropna(axis='columns')
items=pd.read_csv("items.csv")
items_cat=pd.read_csv("item_categories.csv")
sample_submission=pd.read_csv("sample_submission.csv")
shops=pd.read_csv("shops.csv")

print(sample_submission)
print(train)
print(items)
print(items_cat)
print(shops)

all1 = pd.merge(train,items, on='item_id')
all = pd.merge(all1,items_cat, on='item_category_id')


all.sort_values("date_block_num", inplace = True) 
print(all)

"""
all.set_index("date", inplace=True)
print(all.loc['06.01.2013'])

mois=[]
annee=[]
monthy=[]
for item in all["date"]:
    date,month,year=item.split('.')
    test=str(month)+"-"+str(year)
    monthy.append(test)


all["monthy"]=monthy

all["monthy"] =pd.to_datetime(all.monthy)


#gk = all.groupby(['shop_id',"monthy"])["item_cnt_day"].count()
gk = all.groupby(["date_block_num"])["item_cnt_day"].count()
print(gk)
"""

#as_index ^pur eviter le multiindex
#sum ou count?
gk = all.groupby(["date_block_num","item_id"], as_index=False)["item_cnt_day"].sum()
df = pd.DataFrame(gk)
print (df)

df1 = pd.merge(df,all[['item_id','item_category_id']], how='left',on='item_id')
df1 = df1.drop_duplicates().reset_index(drop = True)
df1 = df1[['date_block_num', 'item_category_id', 'item_id', 'item_cnt_day']]
print(df1)

sample_submission["date_block_num"]=34
sample_submission.rename(columns={'ID':'item_id'}, inplace=True)
del sample_submission['item_cnt_month']
sample_submission = pd.merge(sample_submission,all[['item_id','item_category_id']], how='left',on='item_id')
sample_submission = sample_submission.drop_duplicates().reset_index(drop = True)
sample_submission = sample_submission[['date_block_num', 'item_category_id', 'item_id']]
print(sample_submission)
print(sample_submission.isna().sum())


"""
# plot data
fig, ax = plt.subplots(figsize=(15,7))
# use unstack()
gk.plot(ax=ax)
plt.show()
"""




# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	
	return np.array(X), np.array(y)


y=df1["item_cnt_day"]
X=df1[["item_id","item_category_id","date_block_num"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# convert to [rows, columns] structure
np.set_printoptions(suppress=True)
df1 = df1.to_numpy()

print(df1)


# choose a number of time steps
n_steps = 3
# convert into input/output
X, y = split_sequences(df1, n_steps)
print(X.shape, y.shape)
# summarize the data
"""
for i in range(len(X)):
	print(X[i], y[i])
"""

# define model
model = Sequential()
model.add(LSTM(32, activation='relu', input_shape=(n_steps, 3)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=15, verbose=0)

# make predictions
predict_train = model.predict(X)
print(predict_train)
print(len(predict_train))


# demonstrate prediction

sample_submission=sample_submission.to_numpy()
print(sample_submission)

x_input = sample_submission
yhat = model.predict(x_input, verbose=0)
print(yhat)