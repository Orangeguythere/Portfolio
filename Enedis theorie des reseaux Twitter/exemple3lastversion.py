# -*- coding: utf8 -*-
import networkx as nx 
import pandas as pd
import pdb
import matplotlib.pyplot as plt
import pickle
import math
with open('Data/DataFinalYES','rb') as fichier:
	A=pickle.load(fichier)



Data=A[0]
Information=A[1]

## Creation du graphe
g = nx.Graph()

## Ajout des noeuds
for noeud1 in Data.keys():
	g.add_node(noeud1)

## Ajout des liens
for identifiant in g.nodes(data=True):
	noeud1=identifiant[0]
	for noeud2 in Data[noeud1].keys():
		if noeud1!=noeud2 and Data[noeud1][noeud2]==1 \
		and g.has_edge(noeud1,noeud2)==False:
			g.add_edge(noeud1,noeud2)
			
## Degré des noeuds
Dict_degree_centrality=nx.degree_centrality(g)

## Sélection des noeuds + attributs de taille, couleur et nom
for noeud in Data.keys():
	# On retire les noeuds avec degré<10%
	#if Dict_degree_centrality[noeud]<0.1:
		#g.remove_node(noeud)

	# Attributs des noeuds	

	#else:
	#print(Dict_degree_centrality[noeud])
	g.add_edges_from([(1,2), (2,3), (2,4), (3,4)])
	d = nx.degree(g)
	nx.draw_networkx(g, nodelist=d.keys(), node_size=[v * 100 for v in d.values()])

	"""
	print(Dict_degree_centrality[noeud])
	if Dict_degree_centrality[noeud]>=0.1\
		and Dict_degree_centrality[noeud]<0.25:
		g.node[noeud]={'size':100,'color':'yellow','name':''}

	elif Dict_degree_centrality[noeud]>=0.25\
		and Dict_degree_centrality[noeud]<0.35:
		g.node[noeud]={'size':500,'color':'orange','name':Information[noeud]['name']}

	elif Dict_degree_centrality[noeud]>=0.35\
		and Dict_degree_centrality[noeud]<0.5:
		g.node[noeud]={'size':1000,'color':'red','name':Information[noeud]['name']}

	else:
		g.node[noeud]={'size':1500,'color':'blue','name':Information[noeud]['name']}
	"""

Node_Color=[] 
Node_Size=[] 
Node_Label={}
for noeud in g.node.keys():
	Node_Color.append(g.node[noeud]['color'])
	Node_Size.append(g.node[noeud]['size'])
	Node_Label[noeud]=g.node[noeud]['name']


## On représente différemment les liens
Width=[]
Edge_Color=[]
Edge_Kept=[]
for lien in g.edges(data=True):
	noeud1=lien[0]
	noeud2=lien[1]
	# On n'incorpore pas les liens si les deux noeuds sont reliés à moins de 25% du réseau
	if g.node[noeud1]['size']==100 and g.node[noeud2]['size']==100:
		1
		
	# Attributs des liens
	elif (g.node[noeud1]['size']==100 and g.node[noeud2]['size']!=100)\
		or (g.node[noeud1]['size']!=100 and g.node[noeud2]['size']==100):
		Width.append(0.5)
		Edge_Color.append('black')
		Edge_Kept.append((noeud1,noeud2))
	else:
		Width.append(1.5)
		Edge_Color.append('red')
		Edge_Kept.append((noeud1,noeud2))

# len(g.node.keys()) = au nombre de noeuds
# on prend une valeur = 4*valeur par défaut
N=len(g.node.keys())
kv=(1/math.sqrt(N))*4
p =nx.spring_layout(g,k=kv)
plt.figure(figsize=(50,50))
nx.draw_networkx_nodes(g, pos=p, node_size=Node_Size,node_color=Node_Color, alpha=1)
nx.draw_networkx_edges(g, pos=p,edge_color=Edge_Color,width=Width\
			,edgelist=Edge_Kept, alpha=0.05)
nx.draw_networkx_labels(g, pos=p, labels=Node_Label, font_size=35\
			, font_color='black', alpha=4)

nx.write_gexf(g, "file2.gexf", version="1.2draft")

plt.title('Enedisv3', size=50)
plt.axis('off')
plt.savefig('enedis.pdf', format='pdf')


