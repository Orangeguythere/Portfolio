#EDA Tool
import sweetviz as sv
from ydata_profiling import ProfileReport


#Classic
import os
from tqdm.notebook import tqdm
import time
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
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, confusion_matrix,mean_squared_error,r2_score,classification_report,ConfusionMatrixDisplay, log_loss,mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB



import duckdb
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit import RDLogger



#With Spark

#from pyspark.sql import SparkSession 
"""spark = SparkSession.builder.appName("Pyspark Test").config("spark.memory.offHeap.enabled","true").config("spark.memory.offHeap.size","10g").getOrCreate()
df = spark.read.csv(r'D:\DOWNLOADS\leash-BELKA\test.csv', header=True, inferSchema=True) 
df.show(5,0)
transformed_df = df.groupBy("category").count() 
transformed_df.write.format("parquet").save("output/path")"""


#WIth DuckDB
con = duckdb.connect()


df = con.query(f"""(SELECT *
                        FROM parquet_scan('D:\\DOWNLOADS\\train.parquet')
                        ORDER BY random()
                        LIMIT 30000)
                        UNION ALL
                        (SELECT *
                        FROM parquet_scan('D:\\DOWNLOADS\\train.parquet')
                        ORDER BY random()
                        LIMIT 10000)""").df()

con.close()
print(df)



#Extended-connectivity fingerprints (ECFPs) are a type of molecular fingerprint specifically designed for predicting and analyzing molecular activity and properties.
# Convert SMILES to RDKit molecules

###Visualize some of the original data
"""mol=Draw.MolsToGridImage([Chem.MolFromSmiles(x) for x in df['molecule_smiles'][0:12]], molsPerRow=2, subImgSize=(200,200))
mol.show()"""

# Generate ECFPs
def generate_ecfp(molecule, radius=2, bits=1024):
    if molecule is None:
        return None
    return list(AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=bits))

TARGET = "binds"
df['molecule'] = df['molecule_smiles'].apply(Chem.MolFromSmiles)
df['ecfp'] = df['molecule'].apply(generate_ecfp)

#define conditions
conditions = [
    (df['binds_BRD4'] == 1),
    (df['binds_HSA'] == 1),
    (df['binds_sEH'] == 1)
]

#define results
results = ['BRD4', 'HSA', 'sEH']

#create new column based on conditions in column1 and column2
df['protein_name'] = np.select(conditions, results)
print(df)
# One-hot encode the protein_name
onehot_encoder = OneHotEncoder(sparse_output=False)
protein_onehot = onehot_encoder.fit_transform(df['protein_name'].values.reshape(-1, 1))

# Combine ECFPs and one-hot encoded protein_name
X = [ecfp + protein for ecfp, protein in zip(df['ecfp'].tolist(), protein_onehot.tolist())]
y = df[TARGET].tolist()





# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_tab = pd.DataFrame(X_train)
y_tab = pd.DataFrame(y_train)
y_tab = y_tab.rename(columns={0: "binds"})

X_tab_test = pd.DataFrame(X_test)
y_tab_test = pd.DataFrame(y_test)
y_tab_test = y_tab_test.rename(columns={0: "binds"})

df_Gluon_train=pd.concat([X_tab,y_tab],axis=1)
df_Gluon_test=pd.concat([X_tab_test,y_tab_test],axis=1)

# Create and train the model
model = xgb.XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of the positive class


map_score = average_precision_score(y_test, y_pred_proba)
print(f"Mean Average Precision (mAP): {map_score:.2f}")




"""#Autogluon
label= "binds"
train_data = TabularDataset(df_Gluon_train)
predictor = TabularPredictor(label=label,eval_metric="average_precision").fit(train_data,presets="high_quality")
test_data = TabularDataset(df_Gluon_test)
board=predictor.leaderboard(test_data)
print(board)

"""

#Prediction on real test

test_file = 'D:\\DOWNLOADS\\leash-BELKA\\test.csv'
n=0
start_list=[]
tab = pd.DataFrame(columns=['id', 'binds'])
for chunk in pd.read_csv(test_file, chunksize=100000,iterator=False):
    n+=1

    chunk['molecule'] = chunk['molecule_smiles'].apply(Chem.MolFromSmiles)
    chunk['ecfp'] = chunk['molecule'].apply(generate_ecfp)
    protein_onehot_test = onehot_encoder.fit_transform(chunk['protein_name'].values.reshape(-1, 1))

    X_real = [ecfp + protein for ecfp, protein in zip(chunk['ecfp'].tolist(), protein_onehot_test.tolist())]
    #chunk["protein_name"] = chunk["protein_name"].map({"sEH": 1, "BRD4": 2, "HSA": 3}) 

    #Special AutoGluon
    """real_test_data = TabularDataset(X_real)
    #y_pred = predictor.predict_proba(real_test_data)
    y_pred = model.predict_proba(real_test_data)
    transformdf = pd.DataFrame(y_pred)
    new_data = {"id": chunk["id"].values,"binds": transformdf[1].values}
    df_new_rows = pd.DataFrame(new_data)
    tab = pd.concat([tab, df_new_rows], ignore_index=True)"""
    

    #Own Model
    pred = model.predict_proba(X_real)
    transformdf = pd.DataFrame(pred)
    new_data = {"id": chunk["id"].values,"binds": transformdf[1].values}
    df_new_rows = pd.DataFrame(new_data)
    tab = pd.concat([tab, df_new_rows], ignore_index=True)

    print(n)

tab.to_csv('sample_submission.csv', index = False)






