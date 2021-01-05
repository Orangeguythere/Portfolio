#! /usr/bin/python
# -*- coding: utf-8 -*-

import tweepy
import pickle
import pdb
import time

#consumer_key= 
#consumer_secret=
#access_token=
#access_token_secret=

auth = tweepy.auth.OAuthHandler(consumer_key, consumer_secret) 
auth.secure = True
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
#change pour Data2, à la place de Data et voir i + en bas
try:
	with open('Data2','rb') as fichier:
		Base=pickle.load(fichier)

	print('on recommence la collecte')
	List_Followers_enedis_bretagne=Base['enedis_bretagne']['List_Followers']

except:
	print('on commence la collecte pour la première fois')
	# Trouver le nombre de follower de enedis_bretagne"
	user_information=api.get_user(screen_name="@enedis_bretagne")
	nbre_followers_enedis_bretagne=user_information.followers_count

	# Trouver les identifiants des followers de @enedis_bretagne
	List_Followers_enedis_bretagne=[]
	for iden in tweepy.Cursor(api.followers_ids,screen_name="@enedis_bretagne").items(nbre_followers_enedis_bretagne):
		List_Followers_enedis_bretagne.append(iden)


	Base={}
	Base['enedis_bretagne']={}
	Base['enedis_bretagne']['List_Followers']=List_Followers_enedis_bretagne
	


# Pour chaque follower du compte @enedis_bretagne, on récolte son nom, surnom (screen name), la liste de ses followers, la liste de ses friends
#i =0 normalemnt
i=2298
for iden in List_Followers_enedis_bretagne:
	if iden in Base['enedis_bretagne'].keys():
		i=i+1
		print(i)

	else:
		try:	
			A={}

			user_information=api.get_user(id=iden)
			A['name']=user_information.name
			A['screen_name']=user_information.screen_name
			A['created_at']=user_information.created_at

			A['nb_followers']=user_information.followers_count
			A['nb_friends']=user_information.friends_count


			sn=A['screen_name']
			fol=A['nb_followers']
			fr=A['nb_friends']

			List_Followers=[]
			for iden_fol in tweepy.Cursor(api.followers_ids,screen_name=sn).items(fol):
				List_Followers.append(iden_fol)
		
			A['List_Followers']=List_Followers
		

			List_Friends=[]
			for iden_fr in tweepy.Cursor(api.friends_ids,screen_name=sn).items(fr):
				List_Friends.append(iden_fr)

			A['List_Friends']=List_Friends
			

			
			Base['enedis_bretagne'][iden]=A
			Base['enedis_bretagne'][iden]['Problem']='non'


		except Exception as exc:
			print(iden,repr(exc))
			Base['enedis_bretagne'][iden]={}
			Base['enedis_bretagne'][iden]['Problem']='oui'
			


		i=i+1
		with open('Data2','wb') as fichier:
			pickle.dump(Base,fichier)
	
		print(i,len(List_Followers_enedis_bretagne))
		print(api.rate_limit_status()['resources']['followers']['/followers/ids'])
		print(api.rate_limit_status()['resources']['friends']['/friends/ids'])

