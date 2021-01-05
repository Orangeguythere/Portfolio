# -*- coding: utf-8 -*-

import pandas as pd 
import requests
import re
import os
import time
import datetime
import webbrowser

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains



def myselenium():

    req = requests.get("https://www.welcometothejungle.com/fr", timeout=10)
    print(req.status_code)
    
    # Chromedriver 
    options = Options()
    options.headless = True

    driver = webdriver.Chrome(executable_path="/home/linux/Documents/Projets Python/Welcometojungle  Bot/chromedriver",chrome_options=options)
 
    x=1

    while x <= Nbr_pages:
        driver.get("https://www.welcometothejungle.com/fr/jobs?query=data&page="+str(x)+"&refinementList%5Bcontract_type_names.fr%5D%5B%5D=Stage")
        time.sleep(3)
        print("Page "+str(x))
        with open("Donnes"+str(x)+".html","w",encoding='utf8') as outfile:
                    outfile.write(driver.page_source)   
        x+=1

    driver.close()
    

    


#Start

Dtime = datetime.datetime.now()
Dtime = Dtime.strftime("%x")
Dtime = Dtime.replace("/","-")

#Def le nbr de pages à lire
Nbr_pages=5

myselenium()

x=1
liste=[]
while x<=Nbr_pages:
    with open("Donnes"+str(x)+".html","r",encoding='utf8') as infile:
        text=infile.read()

    nbr_results=re.findall("<h1 class=\"sc-12bzhsi-3 kaJlvc\"><span>(.{1,100}) jobs pour votre recherche</span></h1></header>",text)
    job=re.findall("href=\"/(.{1,100})/companies/(.{1,100})/jobs/(.{1,100})\"><h3 class=",text)
    link=re.findall("href=\"(.{1,500})\"><h3 class=",text)

    y=0
    for item in job:
        try:
            liste.append({"Company":item[1].capitalize(),"Job":item[2].replace("-"," "),"Link":"https://www.welcometothejungle.com"+str(link[y]),"Date":Dtime})
            
        except:
            print("Nope")
        y+=1
            
        
    os.remove("Donnes"+str(x)+".html")
    x+=1

df = pd.DataFrame(liste)
df.drop_duplicates(subset ="Link", keep = "first", inplace = True) 
print (df)


try:
    old=pd.read_json ('old.json')
    print("Old file loaded")

except:
    df.to_json('old.json')
    print("New file created")


newjobs=0
print("List of new jobs:")
for item in df["Job"]:
    if item not in old["Job"].tolist():
        y=df[df['Job'].str.match(item)]
        print(y["Company"].values)
        print(y["Job"].values)
        print(y["Link"].values)
        print("-"*50)
        newjobs+=1


print("*"*100)
print("List of old jobs removed:")
for item in old["Job"]:
    if item not in df["Job"].tolist():
        z=old[old['Job'].str.match(item)]
        print(z["Company"].values)
        print(z["Job"].values)
        print("-"*50)


print("There is "+str(newjobs)+" new jobs since last time.")



Otime = (old["Date"][0]).strftime("%x")
Otime = Otime.replace("/","-")

if Dtime != Otime:
    df.to_json("old.json")
    print("Update date done. Old date was : "+str(Otime))
else :
    print("Update done today.")


        


