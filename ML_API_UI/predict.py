import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


import seaborn as sns
from lazypredict.Supervised import LazyClassifier,LazyRegressor
from sklearn.model_selection import train_test_split

import joblib

def score_classifier(dataset,classifier,labels):

    """
    performs 3 random trainings/tests to build a confusion matrix and prints results with precision and recall scores
    :param dataset: the dataset to work on
    :param classifier: the classifier to use
    :param labels: the labels used for training and validation
    :return:
    """

    kf = KFold(n_splits=3,random_state=50,shuffle=True)
    confusion_mat = np.zeros((2,2))
    recall = 0
    for training_ids,test_ids in kf.split(dataset):
        training_set = dataset[training_ids]
        training_labels = labels[training_ids]
        test_set = dataset[test_ids]
        test_labels = labels[test_ids]
        classifier.fit(training_set,training_labels)
        predicted_labels = classifier.predict(test_set)
        confusion_mat+=confusion_matrix(test_labels,predicted_labels)
        recall += recall_score(test_labels, predicted_labels)
    recall/=3
    print(confusion_mat)
    print(recall)


# Load dataset
df = pd.read_csv(".\\nba_logreg.csv")

# extract names, labels, features names and values
names = df['Name'].values.tolist() # players names
labels = df['TARGET_5Yrs'].values # labels
paramset = df.drop(['TARGET_5Yrs','Name'],axis=1).columns.values
df_vals = df.drop(['TARGET_5Yrs','Name'],axis=1).values

# replacing Nan values (only present when no 3 points attempts have been performed by a player)
for x in np.argwhere(np.isnan(df_vals)):
    df_vals[x]=0.0

# normalize dataset
X = MinMaxScaler().fit_transform(df_vals)

#example of scoring with support vector classifier
score_classifier(X,SVC(),labels)

# TODO build a training set and choose a classifier which maximize recall score returned by the score_classifier function




#EDA Exploratory data analysis  & 
"""analyze_report = sv.analyze(df)
analyze_report.show_html()"""

# 11 missing values pour 3P%
# Je constate aussi que les valeurs en % ne correspondent pas exactement pour les 3PA, mais aussi pour les autres colonnes en %

df["3P%_corrected"]= (df["3P Made"] *100)/df["3PA"]
df["FG_corrected"]= (df["FGM"] *100)/df["FGA"]
df["FT_corrected"]= (df["FTM"] *100)/df["FTA"]

#Mais etant donné que je n'ai pas d'autres informations sur ce dataset, je garde les colonnes initiales et je remplace uniquement les 11 valeurs manquantes par celles calculées (0).
df["3P%"] = df["3P%"].fillna(0)

#En plus de ça, d'aprés la matrice de corrélation (sweetviz), les 3P n'ont pas vraiment d'influence sur la TARGET.

#Data distribution : normal
#Outliers : mis à part les valeur importantes, non

#1st run : Toutes les colonnes
#2nd run : Pour simplifier le modele, garder seulement les colonnes qui n'ont pas de correlations entre elles (24 to 13)

df = df[['Name', 'GP','MIN','PTS','FG%','FT%','REB','AST','STL','BLK','TOV','TARGET_5Yrs']]

#3rd run : Pour un modele où l'utilisateur complete certains parametres pour predire un futur joueur, on garde seulement 3 parametres importants (avec la TARGET):
#D'autres possibilite peuvent etre envisagé, mais si on reste sur des données générales, ces données sont parmis les plus importantes.
df = df[['GP','MIN','PTS','TARGET_5Yrs']]

"""#Fast ML models whitout hyperparameters tunning
reg = LazyClassifier(verbose=0, ignore_warnings=False, custom_metric=recall_score)
#X_train, X_valid, y_train, y_valid)
models, predictions = reg.fit(X_train, X_test,y_train,y_test)
print(models)"""

#SVC Reste un tres bon classifier si on regarde le recall score
X=df.drop(['TARGET_5Yrs'],axis=1)
y=df['TARGET_5Yrs']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#Create a svm Classifier
clf = SVC(kernel='linear',probability=True)

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# New observation test
new_observation = [[3, 5, 0]]
# View predicted probabilities
# A souligner que le SVC predict proba est une approximation differnete de predict, deja basé sur un kf5
new=clf.predict_proba(new_observation)
#print(new[0][1])

#Recall
print("Recall:",recall_score(y_test, y_pred))


#A rajouter pour l'API
#ROC Curve
# xAI with Shap values
"""sample= 0
explainer = shap.TreeExplainer(model, X_train)
shap_values = explainer(X_train)
shap.plots.bar(shap_values)
#shap.plots.scatter(shap_values[:, "year"], color=shap_values[:,"engine"])
#shap.plots.beeswarm(shap_values)"""


#save the model to a file
joblib.dump(clf, 'SVC_model.joblib')
print("Model saved")


