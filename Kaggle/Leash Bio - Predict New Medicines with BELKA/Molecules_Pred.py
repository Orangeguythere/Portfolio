#EDA Tool
import sweetviz as sv
from ydata_profiling import ProfileReport


#Classic
import os
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
                        FROM parquet_scan('D:\\DOWNLOADS\\leash-BELKA\\train.parquet')
                        WHERE binds = 0
                        ORDER BY random()
                        LIMIT 30000)
                        UNION ALL
                        (SELECT *
                        FROM parquet_scan('D:\\DOWNLOADS\\leash-BELKA\\train.parquet')
                        WHERE binds = 1
                        ORDER BY random()
                        LIMIT 30000)""").df()

con.close()
print(df)

###Visualize some of the original data
"""mol=Draw.MolsToGridImage([Chem.MolFromSmiles(x) for x in df['molecule_smiles'][0:12]], molsPerRow=2, subImgSize=(200,200))
mol.show()"""


#MODEL FROM KAGGLE TEST1
#Extended-connectivity fingerprints (ECFPs) are a type of molecular fingerprint specifically designed for predicting and analyzing molecular activity and properties.
# Convert SMILES to RDKit molecules
df['molecule'] = df['molecule_smiles'].apply(Chem.MolFromSmiles)

# Generate ECFPs
def generate_ecfp(molecule, radius=2, bits=1024):
    if molecule is None:
        return None
    return list(AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=bits))

df['ecfp'] = df['molecule'].apply(generate_ecfp)




# One-hot encode the protein_name
onehot_encoder = OneHotEncoder(sparse_output=False)
protein_onehot = onehot_encoder.fit_transform(df['protein_name'].values.reshape(-1, 1))

# Combine ECFPs and one-hot encoded protein_name
X = [ecfp + protein for ecfp, protein in zip(df['ecfp'].tolist(), protein_onehot.tolist())]
y = df['binds'].tolist()

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the random forest model
rf_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]  # Probability of the positive class

# Calculate the mean average precision
map_score = average_precision_score(y_test, y_pred_proba)
print(f"Mean Average Precision (mAP): {map_score:.2f}")



# Process the test.parquet file chunk by chunk
test_file = 'D:\\DOWNLOADS\\leash-BELKA\\test.csv'
output_file = 'submission.csv'  # Specify the path and filename for the output file

# Read the test.parquet file into a pandas DataFrame
for df_test in pd.read_csv(test_file, chunksize=100000):

    # Generate ECFPs for the molecule_smiles
    df_test['molecule'] = df_test['molecule_smiles'].apply(Chem.MolFromSmiles)
    df_test['ecfp'] = df_test['molecule'].apply(generate_ecfp)

    # One-hot encode the protein_name
    protein_onehot = onehot_encoder.transform(df_test['protein_name'].values.reshape(-1, 1))

    # Combine ECFPs and one-hot encoded protein_name
    X_test = [ecfp + protein for ecfp, protein in zip(df_test['ecfp'].tolist(), protein_onehot.tolist())]

    # Predict the probabilities
    probabilities = rf_model.predict_proba(X_test)[:, 1]

    # Create a DataFrame with 'id' and 'probability' columns
    output_df = pd.DataFrame({'id': df_test['id'], 'binds': probabilities})

    # Save the output DataFrame to a CSV file
    output_df.to_csv(output_file, index=False, mode='a', header=not os.path.exists(output_file))
