
import pyshark

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
from sklearn.ensemble import RandomForestRegressor
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

"""#df = pd.read_csv(r'D:\DOWNLOADS\labelled_training_data.csv')
df = pd.read_csv(r'D:\DOWNLOADS\shark_data.csv')
print(df)"""

#EDA Exploratory data analysis
"""analyze_report = sv.analyze(df)
analyze_report.show_html()

profile = ProfileReport(df, title="Profiling Report")
profile.to_file("Test_report.html")"""


token = os.environ.get("INFLUXDB_TOKEN")
org = "Nope"
url = "http://localhost:8086"

write_client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)

client = influxdb_client.InfluxDBClient(
    url=url,
    token=token,
    org=org
)


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


bucket="Pyshark"
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
    












