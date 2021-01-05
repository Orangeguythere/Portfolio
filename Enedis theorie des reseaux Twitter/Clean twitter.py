#! /usr/bin/python
# -*- coding: utf-8 -*-


import tweepy
import pickle
import pdb
import time
import csv


Data=[]
l=1
i=0
t=0
d=0
while l<=2:
	if l==1:
		x="Data"
	else:
		x="Data2"

	with open(x,'rb') as fichier:
		Base=pickle.load(fichier)
		for iden in Base['enedis_bretagne']:
			try:
				Liste = (Base['enedis_bretagne'][iden]['screen_name'],#0
				Base['enedis_bretagne'][iden]['name'],#1
				Base['enedis_bretagne'][iden]['List_Friends'],#2
				Base['enedis_bretagne'][iden]['nb_friends'],#3
				Base['enedis_bretagne'][iden]['List_Followers'],#4
				Base['enedis_bretagne'][iden]['nb_followers'],#5
				Base['enedis_bretagne'][iden]["created_at"])#6
				Data.append(Liste)
				t=t+1

			except:
				i=i+1
				continue
	l=l+1


#1er colonne est le follower, le 2nd la catégorie d'infos
print(Data[0][0])
print(Data[0][3])
print(Data[0][2])
print("Nbr qui marche:"+str(t))
print("Erreurs "+str(i))

with open('DataFinalv2','wb') as fichier:
	pickle.dump((Data),fichier)

#Data csv 
with open("data.csv", "w",encoding='utf-8') as outfile:
	datap=csv.writer(outfile,delimiter=';',lineterminator='\n')
	datap.writerow(["Screen name","Name","List friends","Nb List friends","List_Followers","Nb List_Followers","Date creation"])
	for iden, elements in enumerate(Data):
		Col1=Data[iden][0]
		Cola=Data[iden][1]
		Col2=Data[iden][2]
		Col3=Data[iden][3]
		Col4=Data[iden][4]
		Col5=Data[iden][5]
		Col6=Data[iden][6]
		datap.writerow([Col1,Cola,Col2,Col3,Col4,Col5,Col6])
"""





Base = {}


with open('Data','rb') as fichier:
    Base.update(pickle.load(fichier))   
with open('Data2','rb') as fichier:
    Base.update(pickle.load(fichier))   
print (Base)

"""


"""
# On verifie que dans la base les listes de followers et de friends contiennent tous les followers et les friends - si les écarts sont trop grands  (on va considérer un écart supérieur à 10%) on les supprime
for iden in Base['enedis_bretagne'].keys():
	if iden!='List_Followers' and Base['enedis_bretagne'][iden]['Problem']=='non':
		fr_obs=len(Base['enedis_bretagne'][iden]['List_Friends'])
		fr_esp=Base['enedis_bretagne'][iden]['nb_friends']
		fol_obs=len(Base['enedis_bretagne'][iden]['List_Followers'])
		fol_esp=Base['enedis_bretagne'][iden]['nb_followers']

		if fr_obs==fr_esp and fol_obs==fol_esp:
			1
		else:

			if 0.95<=fr_obs/fr_esp<=1.05 and 0.90<=fol_obs/fol_esp<=1.10:
				print('probleme minime',iden,'&',fol_obs,fol_esp,'&',fr_obs,fr_esp)
			else:
				print('probleme important',iden,'&',fol_obs,fol_esp,'&',fr_obs,fr_esp)
				print('on supprime identifiant ', iden)
				del Base['enedis_bretagne'][iden] 
				Base['enedis_bretagne'][iden]={}
				Base['enedis_bretagne'][iden]['Problem']='oui'
			print('\n')


# On calcule un indicateur de réussite dans la collecte
List_Good=[]
List_No_Good=[]
for iden in Base['enedis_bretagne'].keys():
	if iden!='List_Followers' and Base['enedis_bretagne'][iden]['Problem']=='non':
		List_Good.append(iden)
	if iden!='List_Followers' and Base['enedis_bretagne'][iden]['Problem']=='oui':
		List_No_Good.append(iden)


print('On a réussi à collecter ', len(List_Good) ,'followers de enedis_bretagne')
print('Pour ', len(List_No_Good) ,'followers de enedis_bretagne, il y a eu un pb dans la collecte')


# On construit la base de données finale
# Data['1234']['2345']=1 signifie que '2345' est un follower de '1234', Data['1234']['2345']=0 signifie que '2345' n'est pas un follower de '1234' 
# Par convention : on considère Data['1234']['1234']=0
Data={}
for iden1 in List_Good:
	Data[iden1]={}
	for iden2 in List_Good:
		if iden2!=iden1:
			if iden2 in Base['enedis_bretagne'][iden1]['List_Followers']:
				Data[iden1][iden2]=1
			else:
				Data[iden1][iden2]=0
		else:
			Data[iden1][iden2]=0

# On enregistre les informations spécifiques à chaque individu (name, screen_name, date de création du compte...)
Information={}
for iden in List_Good:
	Information[iden]=Base['enedis_bretagne'][iden]


with open('DataFinal','wb') as fichier:
	pickle.dump((Data,Information),fichier)
	
"""