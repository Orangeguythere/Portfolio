#EDA Tool
import sweetviz as sv
from ydata_profiling import ProfileReport


#Classic
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
from autogluon.tabular import TabularDataset, TabularPredictor
from lazypredict.Supervised import LazyClassifier,LazyRegressor


#Sklearn 
from sklearn.model_selection import train_test_split,RepeatedKFold,KFold,StratifiedKFold, cross_val_score,GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, recall_score, fbeta_score, confusion_matrix,mean_squared_error,r2_score,classification_report,ConfusionMatrixDisplay, log_loss,mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer




# Identification de l'interface MLflow
# COMMAND TO START INTERFACE IN CMD : python -m mlflow ui
server_uri = 'http://127.0.0.1:5000'
mlflow.set_tracking_uri(server_uri)
mlflow.set_experiment('CirrhosisV3')


df = pd.read_csv(r'C:\Users\maxim\Projets-Coding\Kaggle\Cirrhosis Patient Survival Prediction\cirrhosis.csv')
print(df)


#EDA Exploratory data analysis
"""analyze_report = sv.analyze(df)
analyze_report.show_html()

profile = ProfileReport(df, title="Profiling Report")
profile.to_file("Test_report.html")
"""
df["Drug"] = df["Drug"].map({"Placebo": 0, "D-penicillamine": 1})
df["Sex"] = df["Sex"].map({"M": 0, "F": 1})
df["Edema"] = df["Edema"].map({"N": 0, "S": 1, "Y": 1})

#Drop all NA row
#df = df.dropna() 


#Drop duplicates
show_dup= df.duplicated().sum()
df.drop_duplicates(inplace=True) 



#Encoding
enc = OrdinalEncoder()
enc.fit(df[["Ascites","Hepatomegaly", "Spiders"]])
df[["Ascites","Hepatomegaly", "Spiders"]] = enc.transform(df[["Ascites","Hepatomegaly", "Spiders"]])

#Put every column in FLOAT64
#print(df.dtypes)
#df = df.astype({"year":'float', "selling_price":'float',"km_driven":'float',"engine":'float'})  


#Outliers : Best and worse exemple in df

#PREDICTION
TARGET = "Status"
FEATURES = [col for col in df.columns if col not in ["id", TARGET]]

#Standardisation (no need if tree based algo)
"""scaler = StandardScaler()
df[FEATURES]= scaler.fit_transform(df[FEATURES])
"""
y = df[TARGET].map({"C": 0, "CL": 1, "D": 2}) 
X = df[FEATURES]


mlflow.autolog()
mlflow.start_run()



space = {
    'objective': 'multi_logloss', 
    'max_depth': 9, 
    'learning_rate': 0.034869481921747415, 
    'n_estimators': 10000,
    'min_child_weight': 9, 
    'colsample_bytree': 0.2, 
    'reg_alpha': 0.10626128775335533, 
    'reg_lambda': 0.624196407787772, 
    'random_state': 42
}
 # Classic Train-test split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

model =  xgb.XGBClassifier(**space)

my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer(strategy='most_frequent')),
                              ('clf', model)
                             ])



# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

print("Mean Square Error: " + str(mean_squared_error(preds, y_valid)))


mlflow.end_run()




"""
params={
    'clf__n_estimators':[500,1000],
    'clf__max_depth': [5, 6, 7, 8]}

grid_pipe = GridSearchCV(my_pipeline,
                         param_grid=params,
                         cv=5,
                         verbose=1)

grid_pipe.fit(X_train, y_train)
print(grid_pipe.best_params_)
print(grid_pipe.best_score_)"""







"""#Pred for submissions.csv
Real_test = pd.read_csv(r'D:\DOWNLOADS\test.csv')
#sub_file = pd.read_csv(r'D:\DOWNLOADS\sample_submission.csv')
test=Real_test

test["Drug"] = test["Drug"].map({"Placebo": 0, "D-penicillamine": 1})
test["Sex"] = test["Sex"].map({"M": 0, "F": 1})
test["Edema"] = test["Edema"].map({"N": 0, "S": 1, "Y": 1})

#Encoding
enc = OrdinalEncoder()
enc.fit(test[["Ascites","Hepatomegaly", "Spiders"]])
test[["Ascites","Hepatomegaly", "Spiders"]] = enc.transform(test[["Ascites","Hepatomegaly", "Spiders"]])


ids=[]
Status_C=[]
Status_CL=[]
Status_D=[]
test = test.drop('id',axis = 1)
i=0

pred = my_pipeline.predict_proba(test)

for id in Real_test["id"]:
    
    #print(id)  
    ids.append(id)
    Status_C.append(pred[i][0])
    Status_CL.append(pred[i][1])
    Status_D.append(pred[i][2])
    i+=1


list_of_tuples = list(zip(ids, Status_C,Status_CL,Status_D))
sub_result = pd.DataFrame(list_of_tuples, columns=['id', 'Status_C','Status_CL','Status_D'])
print(sub_result)
sub_result.to_csv('sample_submission.csv', index = False)



"""





