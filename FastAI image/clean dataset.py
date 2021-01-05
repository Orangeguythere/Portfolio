# -*- coding: utf-8 -*-
# The above encoding declaration is required and the file must be saved as UTF-8

import pandas as pd
from pandas import concat
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.api as sp
import seaborn as sns
import os
import shutil 

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



data = pd.read_csv("/media/linux/Disque Dur/DOWNLOADS/skin-cancer-mnist-ham10000/HAM10000_metadata.csv")
print(data)

groupes = data.groupby("dx").first()
groupes=groupes.index.tolist()
print(groupes)

try : 
    for groupe in groupes:
        os.makedirs("/media/linux/Disque Dur/DOWNLOADS/skin-cancer-mnist-ham10000/"+groupe)
except :
    print("groupe alreeady created")

n=0
"""
for item in data["image_id"]:
    try :
        print(item)
        print(data["dx"][n])
        filePath="/media/linux/Disque Dur/DOWNLOADS/skin-cancer-mnist-ham10000/ham10000_images_part_1/"+item+".jpg"
        folderPath="/media/linux/Disque Dur/DOWNLOADS/skin-cancer-mnist-ham10000/"+data["dx"][n]
        shutil.copy(filePath, folderPath)
        n+=1
        
    except : 
        n+=1
        print("Other folder")
"""

for item in data["image_id"]:
    try :
        print(item)
        print(data["dx"][n])
        filePath="/media/linux/Disque Dur/DOWNLOADS/skin-cancer-mnist-ham10000/ham10000_images_part_2/"+item+".jpg"
        folderPath="/media/linux/Disque Dur/DOWNLOADS/skin-cancer-mnist-ham10000/"+data["dx"][n]
        shutil.copy(filePath, folderPath)
        n+=1
        
    except : 
        n+=1
        print("Other folder")