
import pyshark
import ast
import json
import logging
import re 


import influxdb_client, os, time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

#EDA Tool
import sweetviz as sv
from ydata_profiling import ProfileReport


#Classic
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as ltb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
from lime import lime_tabular

#MLops
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval 
from hyperopt.early_stop import no_progress_loss
import optuna
from optuna.samplers import TPESampler
from autogluon.tabular import TabularDataset, TabularPredictor
from lazypredict.Supervised import LazyClassifier,LazyRegressor

#Sklearn 
from sklearn.model_selection import train_test_split,RepeatedKFold,KFold,StratifiedKFold, cross_val_score,GridSearchCV
from sklearn.ensemble import RandomForestRegressor,IsolationForest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder, StandardScaler,OneHotEncoder,MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix,mean_squared_error,r2_score,classification_report,ConfusionMatrixDisplay, log_loss,mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB


# Identification de l'interface MLflow
# COMMAND TO START INTERFACE IN CMD : python -m mlflow ui
"""server_uri = 'http://127.0.0.1:5000'
mlflow.set_tracking_uri(server_uri)
mlflow.set_experiment('CiorrhosisV2')"""



df = pd.read_csv(r'C:\Users\maxim\Projets-Coding\Kaggle\cyber_FINAL.csv')
print(df.head(30))
print(df.dtypes)


"""#EDA Exploratory data analysis
analyze_report = sv.analyze(df)
analyze_report.show_html()

profile = ProfileReport(df, title="Profiling Report")
profile.to_file("Test_report.html")

"""





#Isolation Forest
FEATURES = [col for col in df.columns if col not in ['userId','stackAddresses','Value']]


#Own preprocessing 
#Encoding
enc = OrdinalEncoder()
enc.fit(df[["processName","hostName", "eventName","Name","Type"]])
df[["processName","hostName", "eventName","Name","Type"]] = enc.transform(df[["processName","hostName", "eventName","Name","Type"]])
#df = df.astype({"year":'float', "selling_price":'float',"km_driven":'float',"engine":'float'})  
numerical_cols = [cname for cname in df.columns if df[cname].dtype in ['int64']]
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
df[numerical_cols] = df[numerical_cols].astype('float64')
df=df.dropna() 
#df=df.sample(frac=0.1, replace=True, random_state=42)
print(df.dtypes)



y = df["sus"]
X = df[FEATURES]

#Scaler
scaler = MinMaxScaler()
df= scaler.fit_transform(df[FEATURES])


# Classic Train-test split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

"""
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in X_train.columns if X_train[cname].nunique() < 10 and 
                        X_train[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='mean')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=np.nan,strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


"""

#The predictions are -1 for outlier and 1 for inlier.
model = IsolationForest(max_samples='auto',bootstrap=False, n_jobs=-1, random_state=42)

# Preprocessing of training data, fit model 
model.fit(X_train, y_train)


# Preprocessing of validation data, get predictions
preds = model.predict(X_valid)

# Evaluate the model
y_score = model.decision_function(X)


X_valid['anomaly']=preds
outliers=X_valid.loc[X_valid['anomaly']==-1]
outlier_index=list(outliers.index)
#Number of anomalies and normal points here points classified -1 are anomalous
print(X_valid['anomaly'].value_counts())



#Graphics Isoltation forest
# Compute y_score using X_valid[FEATURES] instead of X_valid
y_score = model.decision_function(X_valid[FEATURES])

# Sort the data by the anomaly score
sorted_index = np.argsort(y_score)
y_score_sorted = y_score[sorted_index]
anomaly_sorted = X_valid['anomaly'].iloc[sorted_index]

# Plot the data, coloring anomalies red
plt.scatter(range(len(y_score_sorted)), y_score_sorted, c=anomaly_sorted, cmap='viridis')
plt.xlabel('Data point index')
plt.ylabel('Anomaly score')
plt.title('Isolation Forest Anomaly Detection')
plt.colorbar(label='Anomaly (-1) / Normal (1)')
plt.show()

# Compute y_score using X_valid[FEATURES] instead of X_valid
y_score = model.decision_function(X_valid[FEATURES])

# Create a parallel coordinates plot of the features
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_valid['hostName'], X_valid['eventName'], X_valid['Type'], c=X_valid['anomaly'], cmap='viridis')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
plt.title('Isolation Forest Anomaly Detection')
plt.show()


# xAI with Shap values
explainer = shap.TreeExplainer(model, X)
shap_values = explainer(X,check_additivity=False)
shap.plots.bar(shap_values)
#shap.plots.scatter(shap_values[:, "year"], color=shap_values[:,"engine"])
shap.plots.beeswarm(shap_values)



#Best ML models whitout hyperparameters tunning
"""reg = LazyClassifier(verbose=0, ignore_warnings=False, custom_metric=accuracy_score)
models, predictions = reg.fit(X_train, X_valid, y_train, y_valid)
print(models)
"""

"""af = df.drop(['userId', 'stackAddresses'], axis=1)
label= "sus"
predictor = TabularPredictor(label=label,eval_metric="accuracy").fit(af)
test_data = TabularDataset(af)
y_pred = predictor.predict(test_data.drop(columns=[label]))
predictor.evaluate(test_data, silent=False)
predictor.leaderboard(test_data)
print(predictor.leaderboard(test_data))
"""










"""# Write script
bucket="New_test"
write_api = client.write_api(write_options=SYNCHRONOUS)

p = influxdb_client.Point("my_measurement").tag("location", "Prague").field("temperature", 25.3)
write_api.write(bucket=bucket, org=org, record=p)

"""



"""bucket="test"

write_api = write_client.write_api(write_options=SYNCHRONOUS)
   
for value in range(5):
  point = (
    Point("measurement1")
    .tag("tagname1", "tagvalue1")
    .field("field1", value)
  )
  write_api.write(bucket=bucket, org="Nope", record=point)
  time.sleep(1) # separate points by 1 second

print("ok")
"""


"""bucket="Pyshark"
write_api = client.write_api(write_options=SYNCHRONOUS)

capture = pyshark.LiveCapture(interface="Wi-Fi 2")
capture.sniff(timeout=50)


for packet in capture.sniff_continuously(packet_count=100):
    point = (
    Point(packet['ip'].dst)
    .tag("LiveCapture")
    .field("IP", 1)
  )
    write_api.write(bucket=bucket, org=org, record=point)
    #print ('Just arrived:', packet)
    
"""











