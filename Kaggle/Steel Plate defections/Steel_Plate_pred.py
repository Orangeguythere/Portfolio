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

df = pd.read_csv(r'D:\DOWNLOADS\playground-series-s4e3\train.csv')
Real_test = pd.read_csv(r'D:\DOWNLOADS\playground-series-s4e3\test.csv')
test = Real_test.drop('id',axis = 1)
print(df)

"""#EDA Exploratory data analysis
analyze_report = sv.analyze(df)
analyze_report.show_html()

profile = ProfileReport(df, title="Profiling Report")
profile.to_file("Test_report.html")
"""

#PREDICTION
start_list = []
tab = pd.DataFrame()
TARGET = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
#TARGET = ['Pastry']
for item in TARGET:
    print(item)
    FEATURES = [col for col in df.columns if col not in ['id', 'Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']]

    #Standardisation/Normalisation 
    #scaler = StandardScaler()
    """scaler = MinMaxScaler()
    df[FEATURES]= scaler.fit_transform(df[FEATURES])
    """

    
    y = df[item]
    X = df[FEATURES]
    


    # Classic Train-test split
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    df_Gluon_train=pd.concat([X_train,y_train],axis=1)
    df_Gluon_test=pd.concat([X_valid,y_valid],axis=1)

    # Select categorical columns with relatively low cardinality (convenient but arbitrary)
    categorical_cols = [cname for cname in X_train.columns if X_train[cname].nunique() < 10 and 
                            X_train[cname].dtype == "object"]

    # Select numerical columns
    numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]


    # Keep selected columns only
    my_cols = categorical_cols + numerical_cols
    X_train = X_train[my_cols].copy()
    X_valid = X_valid[my_cols].copy()


    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy='mean')

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])



    """#Best ML models whitout hyperparameters tunning
    reg = LazyClassifier(verbose=0, ignore_warnings=False, custom_metric=roc_auc_score)
    models, predictions = reg.fit(X_train, X_valid, y_train, y_valid)
    print(models)
    """
    #Autogluon
    label= item
    train_data = TabularDataset(df_Gluon_train)
    predictor = TabularPredictor(label=item,eval_metric="roc_auc").fit(train_data)
    #predictor = TabularPredictor(label=item,eval_metric="roc_auc").fit(train_data)
    test_data = TabularDataset(df_Gluon_test)
    board=predictor.leaderboard(test_data)
    print(board)
    #predictor.evaluate(test_data, silent=True)

    real_test_data = TabularDataset(test)
    y_pred = predictor.predict_proba(real_test_data)
    perf = predictor.evaluate(test_data, auxiliary_metrics=False)
    print(perf)

    """
    #Load Autogluon model
    model_savepath = "AutogluonModels/"+str(item)
    model = TabularPredictor.load(model_savepath, require_version_match=False, verbosity=4)
    real_test_data = TabularDataset(test)
    y_pred = model.predict_proba(real_test_data)
    print(y_pred)
    #y_pred = predictor.predict(test_data)"""

    #Load Autogluon Model
    """binarypred = TabularPredictor.load("AutogluonModels/ag-20240310_100414/")
    y_pred_proba_binary=binarypred.predict(test,model='WeightedEnsemble_L2')
    print(y_pred_proba_binary)
"""
    model =  xgb.XGBClassifier()
    #model = GaussianNB()

    my_pipeline = Pipeline(steps=[('preprocessor',preprocessor),
                                ('clf', model)
                                ])


    # Preprocessing of training data, fit model 
    my_pipeline.fit(X_train, y_train)

    # Preprocessing of validation data, get predictions
    preds = my_pipeline.predict(X_valid)

    # xAI with Shap values
    """sample= 0
    explainer = shap.TreeExplainer(model, X)
    shap_values = explainer(X)
    #shap.plots.bar(shap_values)
    #shap.plots.scatter(shap_values[:, "year"], color=shap_values[:,"engine"])
    #shap.plots.beeswarm(shap_values)"""

    
    auc = roc_auc_score(y_valid, preds)
    print(auc)

    #pred = my_pipeline.predict_proba(test)
    new_data = {item: y_pred[1] }
    tab = tab.assign(**new_data)

    #start_list.append(tab)



#Pred for submissions.csv
Real_test = pd.read_csv(r'D:\DOWNLOADS\playground-series-s4e3\test.csv')
#sub_file = pd.read_csv(r'D:\DOWNLOADS\sample_submission.csv')
print(tab)
print(Real_test["id"])
tab =pd.concat([tab,Real_test["id"]],axis=1)
print(tab)
tab.to_csv('sample_submission.csv', index = False)











