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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, recall_score, fbeta_score, confusion_matrix,mean_squared_error,r2_score,classification_report,ConfusionMatrixDisplay, log_loss 





# Identification de l'interface MLflow
# COMMAND TO START INTERFACE IN CMD : python -m mlflow ui
server_uri = 'http://127.0.0.1:5000'
mlflow.set_tracking_uri(server_uri)
mlflow.set_experiment('CiorrhosisV2')

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

"""one_hot = pd.get_dummies(df['Sex'])
# Drop column Sex as it is now encoded
df = df.drop('Sex',axis = 1)
# Join the encoded df
df = df.join(one_hot)"""


#Drop duplicates
df.drop_duplicates(inplace=True) 
show_dup= df.duplicated().sum()
print(show_dup) 

#Encoding
enc = OrdinalEncoder()
enc.fit(df[["Ascites","Hepatomegaly", "Spiders"]])
df[["Ascites","Hepatomegaly", "Spiders"]] = enc.transform(df[["Ascites","Hepatomegaly", "Spiders"]])

#Put every column in FLOAT64
print(df.dtypes)
#df = df.astype({"year":'float', "selling_price":'float',"km_driven":'float',"engine":'float'})  


#Outliers : Best and worse exemple in df

#PREDICTION
TARGET = "Status"
FEATURES = [col for col in df.columns if col not in ["id", TARGET]]

#Standardisation (no need if tree based algo)
"""scaler = StandardScaler()
df[FEATURES]= scaler.fit_transform(df[FEATURES])
"""
y = df[TARGET].map({"D": 0, "CL": 1, "C": 2}) 
X = df[FEATURES]




 # Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Best ML models whitout hyperparameters tunning
"""reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
print(models)"""

"""
label= TARGET
predictor = TabularPredictor(label=label).fit(df)
test_data = TabularDataset(df)
y_pred = predictor.predict(test_data.drop(columns=[label]))
predictor.evaluate(test_data, silent=True)
predictor.leaderboard(test_data)"""




space = {'max_depth': hp.choice("max_depth", np.arange(1,20,1,dtype=int)),
        'gamma': hp.uniform ('gamma', 0,1),
        'reg_alpha' : hp.uniform('reg_alpha', 0,50),
        'reg_lambda' : hp.uniform('reg_lambda', 10,100),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0,1),
        'min_child_weight' : hp.uniform('min_child_weight', 0, 5),
        'n_estimators': 10000,
        'learning_rate': hp.uniform('learning_rate', 0, .15),
        #'tree_method':'gpu_hist', 
        #'gpu_id': 0,
        'random_state': 42
        }
        



# Enable MLflow autologging
mlflow.autolog()

def model_tuning(space):
    # Train model
    with mlflow.start_run():
        #ml_model = xgb.XGBRegressor()
        ml_model = xgb.XGBRegressor(**space)
        #ml_model = ltb.LGBMRegressor()
        ml_model.fit(X_train, y_train)
        

        # Log model
        #mlflow.log_model(ml_model, 'model')
        #mlflow.shap.autolog()
        predictions = ml_model.predict(X_test)

        
        """# xAI with Shap values
        sample= 0
        explainer = shap.TreeExplainer(ml_model, X)
        shap_values = explainer(X)
        #shap.plots.bar(shap_values)
        #shap.plots.scatter(shap_values[:, "year"], color=shap_values[:,"engine"])
        #shap.plots.beeswarm(shap_values)"""

        """
        id = 0
        sv = explainer(X.loc[[id]])
        exp = shap.Explanation(sv.values[:,:,], sv.base_values[:,], data=X.loc[[id]].values, feature_names=X.columns)
        shap.plots.waterfall(exp[0])"""

        """
        id = 0
        sv = explainer(X.loc[[id]])
        exp = shap.Explanation(sv.values[:,:,], sv.base_values[:,], data=X.loc[[id]].values, feature_names=X.columns)
        shap.force_plot(exp[0],matplotlib=True)"""

        #Lime
        """
        explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.array(X.astype('int')),
        feature_names=X.columns,
        class_names=['0', '1'],
        mode='classification')
        """

        #Metrics
        mse = mean_squared_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions, squared=False)
        R2 = ml_model.score(X_test, y_test)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("R2", R2)

        #log_loss =log_loss(y_test,predictions)
        #mlflow.log_metric("log_loss", log_loss)
        
        #Artefact
        #mlflow.log_figure(cm.figure_, 'test_confusion_matrix.png')

        #Specify what the loss is for each model.
        return {'loss':rmse, 'status': STATUS_OK, 'model': ml_model}

mlflow.end_run()
#Run 20 trials.
trials = Trials()
best = fmin(fn=model_tuning,
            space=space,
            algo=tpe.suggest,
            max_evals=30,
            trials=trials)
print(best)

#Create instace of best model.
best_model = trials.results[np.argmin([r['loss'] for r in 
    trials.results])]['model']


#Examine model hyperparameters
print(best_model)

xgb_preds_best = best_model.predict(X_test)
xgb_score_best = mean_squared_error(y_test, xgb_preds_best, squared=False)
print('RMSE_Best_Model:', xgb_score_best)

xgb_standard = xgb.XGBRegressor().fit(X_train, y_train)
standard_score = mean_squared_error(y_test, xgb_standard.predict(X_test), squared=False)
print('RMSE_Standard_Model:', standard_score)




#Specify what the loss is for each model.



"""
#Pred for submissions.csv
Real_test = pd.read_csv(r'D:\DOWNLOADS\test.csv')
#sub_file = pd.read_csv(r'D:\DOWNLOADS\sample_submission.csv')
test=Clean_dataframe(Real_test)



ids=[]
Status_C=[]
Status_CL=[]
Status_D=[]
test = test.drop('id',axis = 1)
i=0
for id in Real_test["id"]:
    


    print(id)
    
    pred = lr.predict_proba(test)
    ids.append(id)

    Status_C.append(pred[i][0])
    Status_CL.append(pred[i][1])
    Status_D.append(pred[i][2])
    i+=1


list_of_tuples = list(zip(ids, Status_C,Status_CL,Status_D))
sub_result = pd.DataFrame(list_of_tuples, columns=['id', 'Status_C','Status_CL','Status_D'])
print(sub_result)
sub_result.to_csv('sample_submission.csv', index = False)"""

















