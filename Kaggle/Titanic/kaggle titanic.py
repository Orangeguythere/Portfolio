import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.api as sp
import pylab as pl
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
from sklearn import utils
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.metrics import accuracy_score, classification_report



#Data
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

print(train.describe())

#Remplir vide
print (train.isnull().sum())
print (test.isnull().sum())
"""
train["Age"]= train["Age"].fillna(train["Age"].mean())
test["Age"]= test["Age"].fillna(test["Age"].mean())
"""

train["Fare"]= train["Fare"].fillna(train["Fare"].mean())
test["Fare"]= test["Fare"].fillna(test["Fare"].mean())

#Creation colonne prix
col         = 'Fare'
conditions  = [ (train[col]<=10), 
train[col]<=20, 
train[col]<=40,
train[col]<=60,
train[col]<=100,
train[col]<=200,
train[col]>200]
choices     = [1,2,3,4,5,6,7]
train["Fare_cat"] = np.select(conditions, choices)


col         = 'Fare'
conditions  = [ (test[col]<=10), 
test[col]<=20, 
test[col]<=40,
test[col]<=60,
test[col]<=100,
test[col]<=200,
test[col]>200]
choices     = [1,2,3,4,5,6,7]
test["Fare_cat"] = np.select(conditions, choices)

#Remplissage inteligent de l'age vide
tab=train[['Age', 'Fare_cat','Sex']].groupby(['Fare_cat','Sex'], as_index=False).agg(['mean','count'])
print(tab)
print(tab['Age']["mean"][1][0])


n=0
while n < len(train):
    if train["Age"].isnull().iloc[n]:
        if train["Sex"].iloc[n]=="male":
            x=train["Fare_cat"].iloc[n]
            train["Age"].iloc[n]=tab['Age']["mean"][x][1]

        else:
            x=train["Fare_cat"].iloc[n]
            train["Age"].iloc[n]=tab['Age']["mean"][x][0]

    n+=1
    
tab=test[['Age', 'Fare_cat','Sex']].groupby(['Fare_cat','Sex'], as_index=False).agg(['mean','count'])
print(tab)
print(tab['Age']["mean"][1][0])

n=0
while n < len(test):
    if test["Age"].isnull().iloc[n]:
        if test["Sex"].iloc[n]=="male":
            x=test["Fare_cat"].iloc[n]
            test["Age"].iloc[n]=tab['Age']["mean"][x][1]

        else:
            x=test["Fare_cat"].iloc[n]
            test["Age"].iloc[n]=tab['Age']["mean"][x][0]

    n+=1

print (train.isnull().sum())
print (test.isnull().sum())

#Creer dummy
train['Genre']=np.where(train['Sex']=='male',1,0)
test['Genre']=np.where(test['Sex']=='male',1,0)


#Variable multinomiale
col         = 'Embarked'
conditions  = [ (train[col] == 'Q'), 
train[col] == 'S', 
train[col] == 'C']
choices     = [1,2,3]
train["Station"] = np.select(conditions, choices)

col         = 'Embarked'
conditions  = [ (test[col] == 'Q'), 
test[col] == 'S', 
test[col] == 'C']
choices     = [1,2,3]
test["Station"] = np.select(conditions, choices)

col         = 'Age'
conditions  = [ (train[col]<=12), 
train[col]<=15, 
train[col]<=20,
train[col]<=30,
train[col]<=40,
train[col]>40]
choices     = [1,2,3,4,5,6]
train["Age_cat"] = np.select(conditions, choices)

col         = 'Age'
conditions  = [ (test[col]<=12), 
test[col]<=15, 
test[col]<=20,
test[col]<=30,
test[col]<=50,
test[col]>50]
choices     = [1,2,3,4,5,6]
test["Age_cat"] = np.select(conditions, choices)


print(train)
print(test)



#On regarde les stats descriptives
print(train.describe())
print(train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).agg(['mean','count']))
print(train[['Age_cat', 'Survived']].groupby(['Age_cat'], as_index=False).agg(['mean','count']))
print(train[['Fare_cat', 'Survived']].groupby(['Fare_cat'], as_index=False).agg(['mean','count']))
print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).agg(['mean','count']))
print(train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).agg(['mean','count']))
print(train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).agg(['mean','count']))
print(train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).agg(['mean','count']))


"""
#Logit simple attention variable binairee!
#On supprime Parch,Station,Fare car p>0.2

X=train[['Pclass',"Genre","Age_cat","SibSp","Parch","Station","Fare_cat","SibSp","Parch"]]
y=train["Survived"]


# Create one-vs-rest logistic regression object
clf = LogisticRegression(C=1000,random_state=0, solver='liblinear')
# Train model
model = clf.fit(X, y)

# Predict class
y_pred=model.predict(test[['Pclass',"Genre","Age_cat","SibSp","Parch","Station","Fare_cat","SibSp","Parch"]])
print(y_pred)


new=[]
for item in test["PassengerId"]:
    new.append([item,y_pred[item-892]])


final=pd.DataFrame(new, columns = ['PassengerId', 'Survived'])
print(final)
print(final[final["Survived"]==1].count())
final.to_csv('Finalsub.csv', index = False)
"""


X=train[['Pclass',"Genre","Age_cat","SibSp","Parch","Station","Fare_cat","SibSp","Parch"]]
y=train["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


gnb = GaussianNB()
KNN = KNeighborsClassifier(n_neighbors=1)
MNB = MultinomialNB()
BNB = BernoulliNB()
LR = LogisticRegression()
SDG = SGDClassifier()
SVC = SVC()
LSVC = LinearSVC()


# Test predict
gnb.fit(X_train, y_train)
y2_GNB_model = gnb.predict(X_test)
print("GaussianNB Accuracy :", accuracy_score(y_test, y2_GNB_model))

KNN.fit(X_train, y_train)
y2_KNN_model = KNN.predict(X_test)
print("KNN Accuracy :", accuracy_score(y_test, y2_KNN_model))

MNB.fit(X_train, y_train)
y2_MNB_model = MNB.predict(X_test)
print("MNB Accuracy :", accuracy_score(y_test, y2_MNB_model))

BNB.fit(X_train, y_train)
y2_BNB_model = BNB.predict(X_test)
print("BNB Accuracy :", accuracy_score(y_test, y2_BNB_model))

LR.fit(X_train, y_train)
y2_LR_model = LR.predict(X_test)
print("LR Accuracy :", accuracy_score(y_test, y2_LR_model))

SDG.fit(X_train, y_train)
y2_SDG_model = SDG.predict(X_test)
print("SDG Accuracy :", accuracy_score(y_test, y2_SDG_model))

SVC.fit(X_train, y_train)
y2_SVC_model = SVC.predict(X_test)
print("SVC Accuracy :", accuracy_score(y_test, y2_SVC_model))

LSVC.fit(X_train, y_train)
y2_LSVC_model = LSVC.predict(X_test)
print("LSVC Accuracy :", accuracy_score(y_test, y2_LSVC_model))

clf = LogisticRegression(C=1000,random_state=0, solver='liblinear')
model = clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
print("Logit reg Accuracy :", accuracy_score(y_test, y_pred))

clf=RandomForestClassifier(n_estimators=100)
model = clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
print("RF classifier Accuracy :", accuracy_score(y_test, y_pred))


#REAL TEST
X=train[['Pclass',"Genre","Age_cat","SibSp","Parch","Station","Fare_cat","SibSp","Parch"]]
y=train["Survived"]

SVC.fit(X,y)
y_pred = SVC.predict(test[['Pclass',"Genre","Age_cat","SibSp","Parch","Station","Fare_cat","SibSp","Parch"]])


new=[]
for item in test["PassengerId"]:
    new.append([item,y_pred[item-892]])


final=pd.DataFrame(new, columns = ['PassengerId', 'Survived'])
print(final)
print(final[final["Survived"]==1].count())
final.to_csv('Finalsub.csv', index = False)









"""
Types of Logistic Regression:

Binary Logistic Regression: The target variable has only two possible outcomes such as Spam or Not Spam, Cancer or No Cancer.
Multinomial Logistic Regression: The target variable has three or more nominal categories such as predicting the type of Wine.
Ordinal Logistic Regression: the target variable has three or more ordinal categories such as restaurant or product rating from 1 to 5.


# new variable : df['GenreBinaire']=np.where(df['Genre']=='H',1,2)

#modifier variable
df.loc[df['Genre'] =='F', 'GenreBinaire']=0
#Autre méthode
df['First Season'] = np.where(df['First Season'] > 1990, 1, df['First Season'])

#Remplir ligne vide par autre chose
data["Q13"]= data["Q13"].fillna('No Data')

# Conserve que les femmes
df_1 = df[df['Genre']=='F']

# Conserve uniquement la colonne Age pour les femmes
df_2 = df['Age'][df['Genre']=='F']
print(df_2)

#Map une valeur dans nouvelle colonne en cherchant par nom MARCHE
#https://stackoverflow.com/questions/46789098/create-new-column-in-dataframe-with-match-values-from-other-dataframe
#df1['Total2'] = df1['Name'].map(df2.set_index('Name')['Total2'])

#Supprime les dup
data.drop_duplicates(subset ="First Name", keep = False, inplace = True) 

#Merge
all = pd.concat([df, ds], ignore_index=True)



#Pour variable multinomiale
col         = 'Q13'
conditions  = [ (data[col] == 'Au moins une fois par semaine'), 
data[col] == 'Au moins une fois par mois', 
data[col] == 'Une fois tous les 2-3 mois',
data[col] == 'Une fois tous les 4-6 mois',
data[col] == 'Une fois par an', 
data[col] == 'Moins souvent',
data[col] == 'Non']

choices     = [ 0.9,0.75,0.6,0.45,0.3,0.15,0 ] #for MCO
#Faire du ordonné avec ! 
data["Q13 multi"] = np.select(conditions, choices)
print(data)

"""




"""
# Draw a scatter plot
fig, ax = plt.subplots(figsize=(16,10),dpi= 80)    
sns.stripplot(data["Q12Echange"],data["AGE_1"], jitter=0.25, size=8, ax=ax, linewidth=.5)
plt.show()
"""

"""
# Draw Stripplot
df_counts = data.groupby(['Q12Vente', 'AGE_1']).size().reset_index(name='counts')
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)    
sns.stripplot(data.Q12Vente, df_counts.AGE_1, size=df_counts.counts*2, ax=ax)
plt.title('Counts Plot - Size of circle is bigger as more points overlap', fontsize=22)
plt.show()
"""

"""
#Violinplot
X = ['Non','Les 2','Acheteur',"Offreur"]
df_counts = data.groupby(["Q12Logement",'AGE_1']).size().reset_index(name='counts')
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
ax = sns.violinplot(data.Q12Logement, data.AGE_1,inner=None, color=".8",order=X)
print(df_counts.AGE_1)
ax = sns.stripplot(data.Q12Logement, data.AGE_1,order=X)
plt.ylim(0, 100)
plt.xlim(None, None)
plt.show()
"""

"""
#Shcema sous forme de pie chart 
df = data["Q18A"].value_counts()
df.plot(kind='pie', subplots=True, figsize=(8, 8),autopct='%1.1f%%')
plt.title("Site d'échange")
plt.ylabel("")
plt.show()
"""


"""
#Treemap
df = data.groupby('Q12Echange').size().reset_index(name='counts')
labels = df.apply(lambda x: str(x[0]) + "\n (" + str(x[1]) + ")", axis=1)
sizes = df['counts'].values.tolist()
colors = [plt.cm.Spectral(i/float(len(labels))) for i in range(len(labels))]

# Draw Plot
plt.figure(figsize=(6,4), dpi= 80)
squarify.plot(sizes=sizes, label=labels, color=colors, alpha=.8)

# Decorate
plt.title('Treemap of Vechile Class')
plt.axis('off')
plt.show()
"""

"""
#Histogramme global
X = ['Non','Les 2','Acheteur',"Offreur"]
A = (data["Q12Echange"].value_counts())
B = (data["Q12Location"].value_counts())
C = (data["Q12Vente"].value_counts())
D = (data["Q12Covoiturage"].value_counts())
E = (data["Q12Logement"].value_counts())

df = pd.DataFrame(np.c_[A,B,C,D,E], index=X)
ax = df.plot(kind='bar')
ax.legend(["Echange", "Location","Vente","Covoiturage","Logement"])
print(df)
plt.show()
"""


#Représentation visuelle

#type x,y
#plt.scatter(data["AGE_1"], data["Q13 multi"], s=1)
"""
data["PCS_REP"].value_counts().plot(kind='bar')
plt.show()
"""


"""
#MCO Regression lineaire multiple

estimation_ols = sm.ols(formula=' Q("Q12Echange") ~ Q("Genre") + Q("AGE_1") + Q("No_risk") + Q("Parisien")+ Q("Confiant")'
, data=data).fit()

print(estimation_ols.summary())
"""

"""
# LOGIT BINAIRE AVEC Y  0 ET 1
#Si Y = Q12, alors perosnne utilisant un site d'echange 
#Rajouter effet marginales 

estimation_logit = sm.logit(formula='Q("Q12") ~ Q("Genre") + Q("Age categories") + Q("No_risk") + Q("Parisien") + Q("Confiant") + Q("Ouvriers") + Q("Artisans, commera§ants et chefs d\'entreprise") + Q("Cadres et professions intellectuelles superieures") + Q("Retraites") + Q("Lyceen, etudiant")'
, data=data).fit()
# Estimation des coefficients
print(estimation_logit.params)
# Estimation des coefficients et statistiques de test
print(estimation_logit.summary())
# Effet marginales
margeff = estimation_logit.get_margeff()
print(margeff.summary())




#USE SP AND NOT SM

#exog=x and endo=y

y = data["Q12Covoiturage"] # define the target variable (dependent variable) as y
anes_exog =data[['Genre',"No_risk","Parisien","Confiant","Ouvriers","Artisans, commera§ants et chefs d\'entreprise","Cadres et professions intellectuelles superieures","Retraites","Lyceen, etudiant","Age categories","Transport_easy","Benevole","NBRE_PERS_FOYER_1"]]
#Pourquoi ajouter une constante?
anes_exog = sp.add_constant(anes_exog, prepend=False)

mlogit_mod = sp.MNLogit(y, anes_exog)
mlogit_res = mlogit_mod.fit()
print(mlogit_res.params)
print(mlogit_res.summary())
print(mlogit_res.predict())


results_as_html = mlogit_res.summary().tables[1].as_html()
final=pd.read_html(results_as_html, header=0, index_col=0)[0]
final=final.reset_index(drop=False).style.applymap(color_negative_red,subset=['P>|z|'])
final.to_excel("FinalQ12Echange.xlsx")

results_as_html = mlogit_res.summary().tables[0].as_html()
debut=pd.read_html(results_as_html, header=0, index_col=0)[0]
debut.to_excel("DebutQ12Echange.xlsx")

"""

"""
margeff = mlogit_res.get_margeff()
print(margeff.summary())
"""
"""

#Logit simple attention variable binairee!
y = data["Q12Echange"] # define the target variable (dependent variable) as y
anes_exog =data[['Genre',"No_risk","Parisien","Confiant","Ouvriers","Artisans, commera§ants et chefs d\'entreprise","Cadres et professions intellectuelles superieures","Retraites","Lyceen, etudiant","AGE_1","Transport_easy","Benevole","NBRE_PERS_FOYER_1"]]
anes_exog = sp.add_constant(anes_exog, prepend=False)

mlogit_mod = sp.Logit(y, anes_exog)
mlogit_res = mlogit_mod.fit()
print(mlogit_res.params)
print(mlogit_res.summary())

"""

"""
#Version sklearn LOGIT

X=data[['Genre',"No_risk","Parisien","Confiant","Ouvriers","Artisans, commera§ants et chefs d\'entreprise","Cadres et professions intellectuelles superieures","Retraites","Lyceen, etudiant","Age categories","Transport_easy","Benevole","NBRE_PERS_FOYER_1"]]
y=data["Q12Vente"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#x_train, x_test, y_train, y_test

# Standarize features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Create one-vs-rest logistic regression object
clf = LogisticRegression(random_state=0, multi_class='multinomial', solver='newton-cg')
# Train model
model = clf.fit(X_std, y)
# Create new observation
#Etudiant rennais
new_observation = [[1,0,0,1,0,0,0,0,1,1,1,0,1]]
#Vieux retraité paris
#new_observation = [[1,1,1,0,0,0,0,1,0,5,1,0,2]]
# Predict class
model.predict(new_observation)
# View predicted probabilities
print(model.predict_proba(new_observation))
# Use score method to get accuracy of model
score = clf.score(X_test, y_test)
print(score)
predictions = clf.predict(X_test)







#Random forest classifier
X=data[['Genre',"No_risk","Parisien","Confiant","Ouvriers","Artisans, commera§ants et chefs d\'entreprise","Cadres et professions intellectuelles superieures","Retraites","Lyceen, etudiant","Age categories","Transport_easy","Benevole","NBRE_PERS_FOYER_1"]]
y=data["Q12Echange"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=1000)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

feature_imp = pd.Series(clf.feature_importances_,index=X.columns).sort_values(ascending=False)
print(feature_imp)
"""

"""
#Confusion matrix
cm = metrics.confusion_matrix(y_test, predictions)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15)
"""


"""
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Score')
plt.ylabel('Caractéristiques')
plt.title("Visualisation des caractéristiques importantes")
plt.legend()
plt.show()
"""


#Forest  regressor

"""
# Pull out one tree from the forest
tree = clf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = X.columns, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')

# Write graph to a png file
graph.write_png('tree.png')



# Limit depth of tree to 3 levels
rf_small = RandomForestClassifier(n_estimators=10, max_depth = 3)
rf_small.fit(X_train,y_train)
# Extract the small tree
tree_small = rf_small.estimators_[5]
# Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = X.columns, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
graph.write_png('small_tree.png')
"""