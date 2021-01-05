# -*- coding: utf-8 -*-
# The above encoding declaration is required and the file must be saved as UTF-8

import pandas as pd
from pandas import concat
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.api as sp
import seaborn as sns

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


import torch
print(torch.cuda.is_available())

from fastai.vision import *
from fastai.metrics import error_rate



#train = "/home/linux/Documents/Projets Python/Kaggle/FastAI image/mnist_sample"
train = "/media/linux/Disque Dur/DOWNLOADS/skin-cancer-mnist-ham10000/Data done"


tfms = get_transforms(do_flip=False)
#Change size depending quality image
data = ImageDataBunch.from_folder(train, ds_tfms=tfms, size=128)
print(data)
print(data.classes)

data.show_batch(rows=3, figsize=(5,5))
#plt.show()

learn = cnn_learner(data, models.resnet34, metrics=error_rate)

#Reactiver pour creer le model
learn.model
learn.fit_one_cycle(4)

#Save and load
#learn.save('stage-1')
#learn = learn.load("stage-1")

interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()
len(data.valid_ds)==len(losses)==len(idxs)


interp.plot_top_losses(9, figsize=(15,11))
plt.show()

interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
plt.show()

print(interp.most_confused(min_val=2))



"""
#Pour ameliorer 

learn.lr_find()
learn.recorder.plot()
learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))

"""


"""
#Pour predict une image

pred_class,pred_idx,outputs = learn.predict(img)
pred_class

"""