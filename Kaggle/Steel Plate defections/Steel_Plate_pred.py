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

df = pd.read_csv(r'D:\DOWNLOADS\playground-series-s4e3\train.csv')
Real_test = pd.read_csv(r'D:\DOWNLOADS\playground-series-s4e3\test.csv')
test = Real_test.drop('id',axis = 1)


"""#EDA Exploratory data analysis
analyze_report = sv.analyze(df)
analyze_report.show_html()

profile = ProfileReport(df, title="Profiling Report")
profile.to_file("Test_report.html")
"""

#Feature engineering
df['X_Range'] = df['X_Maximum'] - df['X_Minimum']
df['Y_Range'] = df['Y_Maximum'] - df['Y_Minimum']
df['Luminosity_Diff'] = df['Maximum_of_Luminosity'] - df['Minimum_of_Luminosity']

test['X_Range'] = test['X_Maximum'] - test['X_Minimum']
test['Y_Range'] = test['Y_Maximum'] - test['Y_Minimum']
test['Luminosity_Diff'] = test['Maximum_of_Luminosity'] - test['Minimum_of_Luminosity']
print(df)


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
    scaler = MinMaxScaler()
    #df[FEATURES]= scaler.fit_transform(df[FEATURES])
    

    
    y = df[item]
    X = df[FEATURES]
    


    # Classic Train-test split
    #X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    """df_Gluon_train=pd.concat([X_train,y_train],axis=1)
    df_Gluon_test=pd.concat([X_valid,y_valid],axis=1)
    """
    """# Select categorical columns with relatively low cardinality (convenient but arbitrary)
    categorical_cols = [cname for cname in X_train.columns if X_train[cname].nunique() < 10 and 
                            X_train[cname].dtype == "object"]

    # Select numerical columns
    numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]"""


    """# Keep selected columns only
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

    """

    """#Best ML models whitout hyperparameters tunning
    reg = LazyClassifier(verbose=0, ignore_warnings=False, custom_metric=roc_auc_score)
    models, predictions = reg.fit(X_train, X_valid, y_train, y_valid)
    print(models)
    """
    """#Autogluon
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
    print(perf)"""

    """
    #Load Autogluon model
    model_savepath = "AutogluonModels/"+str(item)
    model = TabularPredictor.load(model_savepath, require_version_match=False, verbosity=4)
    real_test_data = TabularDataset(test)
    y_pred = model.predict_proba(real_test_data)
    print(y_pred)
    #y_pred = predictor.predict(test_data)"""

    

    def objective(trial):

        params = {
        'grow_policy': trial.suggest_categorical('grow_policy', ["depthwise", "lossguide"]),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
        'gamma' : trial.suggest_float('gamma', 1e-9, 0.5),
        'subsample': trial.suggest_float('subsample', 0.3, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'max_depth': trial.suggest_int('max_depth', 0, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 100.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 100.0, log=True),
        }
    
        """params['booster'] = 'gbtree'
        params['objective'] = 'multi:softmax'
        params["device"] = "cuda"
        params["verbosity"] = 0
        params['tree_method'] = "hist"""
        
        
        #cv_splits = cv.split(train, y=target)
        #cv_scores = list()

        i=0
        n_splits = 10 #5 fold was better
        for train_index, test_index in kf.split(X, y):
            print(i)
            X_train, X_valid = x_scaled[train_index], x_scaled[test_index]
            y_train, y_valid = y[train_index], y[test_index]
        
            model_to_optimise = xgb.XGBClassifier(**params)

            #X_train_fold, X_val_fold = train.iloc[train_idx], train.iloc[val_idx]
            #y_train_fold, y_val_fold = target[train_idx], target[val_idx]

            model_to_optimise.fit(X_train, y_train)

            preds = model_to_optimise.predict_proba(X_valid)
            auc = roc_auc_score(preds, y_valid)
            print(auc)

        
        return auc



    #Setting the kfold parameters
    i=0
    n_splits = 10 #5 fold was better
    kf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 42)
    #kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
    x_scaled = scaler.fit_transform(X)
    for train_index, test_index in kf.split(X, y):
        print(i)
        X_train, X_valid = x_scaled[train_index], x_scaled[test_index]
        y_train, y_valid = y[train_index], y[test_index]


        best_kaggle_xgb_params =  {'n_estimators': 880,
                'learning_rate': 0.015104323232513497,
                'gamma': 0.39584318371982985,
                'reg_alpha': 0.32278378258662743,
                'reg_lambda': 0.9232746535986651,
                'max_depth': 3,
                'min_child_weight': 13,
                'subsample': 0.5038891023587624,
                'colsample_bytree': 0.5216279629980719,
                'random_state': 42
                }
        

        # Activate optuna search by setting run_optimization to True
        run_optimization = False
        if run_optimization:
            sqlite_db = "sqlite:///sqlite.db"
            study_name = "multi_class_prediction_"
            study = optuna.create_study(storage=sqlite_db, study_name=study_name, 
                                        sampler=TPESampler(n_startup_trials=30, multivariate=True, seed=0),
                                        direction="maximize", load_if_exists=True)

            study.optimize(objective, n_trials=100)
            best_cls_params = study.best_params
            best_value = study.best_value
    

            print(f"best optmized accuracy: {best_value:0.5f}")
            print(f"best hyperparameters: {best_cls_params}")

        model =  xgb.XGBClassifier(**best_kaggle_xgb_params)
        model = ltb.LGBMClassifier()


        
        """my_pipeline = Pipeline(steps=[('preprocessor',preprocessor),
                                    ('clf', model)
                                    ])"""


        # Preprocessing of training data, fit model 
        #my_pipeline.fit(X_train, y_train)
        model.fit(X_train, y_train)

        # Preprocessing of validation data, get predictions
        #preds = my_pipeline.predict(X_valid)
        preds = model.predict(X_valid)


        # xAI with Shap values
        """sample= 0
        explainer = shap.TreeExplainer(model, X)
        shap_values = explainer(X)
        #shap.plots.bar(shap_values)
        #shap.plots.scatter(shap_values[:, "year"], color=shap_values[:,"engine"])
        #shap.plots.beeswarm(shap_values)"""

        
        auc = roc_auc_score(preds, y_valid)
        print(auc)

        #Test with real dataset
        test[FEATURES]= scaler.fit_transform(test[FEATURES])
        test = test[FEATURES]

        pred = model.predict_proba(test)
        transformdf = pd.DataFrame(pred)
        new_data = {item+str(i): transformdf[1]}
        tab = tab.assign(**new_data)
        i+=1

        #start_list.append(tab)


print(tab)
x=0
for item in TARGET:
    tab[item]=tab.iloc[:, 0+x: n_splits+x].mean(axis=1)
    x+=n_splits

#Pred for submissions.csv
tab = tab.iloc[:,-7:]
Real_test = pd.read_csv(r'D:\DOWNLOADS\playground-series-s4e3\test.csv')
#sub_file = pd.read_csv(r'D:\DOWNLOADS\sample_submission.csv')
tab =pd.concat([tab,Real_test["id"]],axis=1)
print(tab)
tab.to_csv('sample_submission.csv', index = False)











