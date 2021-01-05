# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd 
import dash_bootstrap_components as dbc
import requests
import re
import os
import time
from threading import Timer
import webbrowser
from dash.dependencies import Input, Output, State
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import plotly.express as px
import plotly.graph_objs as go



## TGV MAX BOOKING SYSTEM ##
starttime=time.time()

def myselenium(depart1,arrive1,date1,refresh,timer):
    
    while True:
        if refresh==0:
            timetowait=1
        else:
            timetowait=timer*60
            
        time.sleep(timetowait - ((time.time() - starttime) % 1))



        req = requests.get("https://www.oui.sncf/", timeout=10)
        print(req.status_code)
    
        depart=depart1
        arrive=arrive1
        date=date1
        #Set HC et date de naissance ici:

        #HC="XXXXXXXXX"
        #naissance="XX/XX/XXXX"
        
        # Chromedriver 
        options = Options()
        options.headless = False

        driver = webdriver.Chrome(executable_path="chromedriver",chrome_options=options)
        driver.get("https://www.oui.sncf/")
        driver.find_element_by_xpath("//*[@id=\"vsb-origin-train-launch\"]").send_keys(depart)
        time.sleep(1)
        driver.find_element_by_xpath("//*[@id=\"vsb-origin-train-launch\"]").send_keys(depart, Keys.ENTER)
        driver.find_element_by_xpath("//*[@id=\"vsb-destination-train-launch\"]").send_keys(arrive)
        time.sleep(1)
        driver.find_element_by_xpath("//*[@id=\"vsb-destination-train-launch\"]").send_keys(arrive, Keys.ENTER)


        driver.find_element_by_xpath("//*[@id=\"vsb-dates-dialog-train-launch-aller-retour-1\"]/div[1]/div/div[1]/span[2]").click()
        driver.find_element_by_xpath("//*[@id=\"vsb-datepicker-departure-date-input\"]").click()
            
        n=0
        while n <= 30:
            driver.find_element_by_xpath("//*[@id=\"vsb-datepicker-departure-date-input\"]").send_keys(Keys.BACKSPACE)
            n+=1


        driver.find_element_by_xpath("//*[@id=\"vsb-datepicker-departure-date-input\"]").send_keys(date)
        time.sleep(1)
        driver.find_element_by_xpath("//*[@id=\"vsb-datepicker-train-launch-aller-retour-submit\"]").click()
        time.sleep(1)

        driver.find_element_by_xpath("//*[@id=\"vsb-passenger_1_train-launch-options-button\"]").click()
        driver.find_element_by_xpath("//*[@id=\"passenger_1_train-launch-discount-card-type\"]").send_keys("TGVmax", Keys.ENTER)
        time.sleep(1)
        driver.find_element_by_xpath("//*[@id=\"passenger_1_train-launch-discount-card-number\"]").click()
        driver.find_element_by_xpath("//*[@id=\"passenger_1_train-launch-discount-card-number\"]").send_keys(HC)
        driver.find_element_by_xpath("//*[@id=\"passenger_1_train-launch-discount-card-dateofbirth\"]").click()
        driver.find_element_by_xpath("//*[@id=\"passenger_1_train-launch-discount-card-dateofbirth\"]").send_keys(naissance)
        driver.find_element_by_xpath("//*[@id=\"vsb-passenger-options-remote-button-confirm\"]/span").click()

        driver.find_element_by_xpath("//*[@id=\"vsb-booking-train-launch-submit\"]").click()

        time.sleep(15)

        driver.execute_script("window.scrollTo(0, 100)") 
        #Pour enlever la page cookies
        time.sleep(10)

        dic=[
        "/html/body/div[1]/div[6]/main/section/div/div[3]/div/div[2]/div/div[12]/span/span",
        "/html/body/div[1]/div[6]/main/section/div/div[3]/div/div[2]/div/div[17]/span/span",
        "/html/body/div[1]/div[6]/main/section/div/div[3]/div/div[2]/div/div[22]/span/span",
        "/html/body/div[1]/div[6]/main/section/div/div[3]/div/div[2]/div/div[27]/span/span"]

        x=0
        Gx=0
        gtaille=0
        
        for item in dic: 

            time.sleep(4)

            try:
                driver.find_element_by_xpath(item).click()
                with open("Donnes"+str(x)+".html","w",encoding='utf8') as outfile:
                    outfile.write(driver.page_source)

                taille = os.stat("Donnes"+str(x)+".html").st_size
                if taille>=gtaille:
                    gtaille=taille
                    Gx=x

                print(Gx)

            except:
                print(item)

            x+=1

            
        time.sleep(5)

        driver.close()

        
        with open("Donnes"+str(Gx)+".html","r",encoding='utf8') as infile:
            text=infile.read()

        
        

        H=re.findall("datetime=\"(.{1,100})\" aria-label",text)
        print(H)
        print(len(H))
        P=re.findall("data-auto=\"DATA_PRICE_BTN_PRICEBTN_SECOND\" data-price=\"(.{1,100})\" class",text)
        print(P)
        print(len(P))
        D=re.findall("class=\"oui-duration-formatter__screen-reader-label___(.{1,100})\">(.{1,100})</span></time>",text)
        print(D)
        print(len(D))
        T=re.findall("class=\"travel-timeline-segment_typeName__3sX9R\">(.{1,100})</span></p>",text)
        print(T)
        print(len(T))
        Trajet=re.findall("data-auto=\"FIELD_SUMMARY_LOCALITY\">(.{1,100})</span></div>", text)
        print(Trajet)
        print(len(Trajet))


        n=0
        Tout=[]
        while n<(len(P)):
            All={"Depart":H[2*n],"Arrive":H[1+2*n],"Prix":P[n],"Duree":D[n][1],"Type":T[n],"Trajet":Trajet[2*n]+","+Trajet[1+2*n]}
            Tout.append(All)
            n+=1

        df = pd.DataFrame(Tout)
        columnsTitles = ["Depart","Arrive","Duree","Prix","Trajet","Type"]
        df = df.reindex(columns=columnsTitles)
        
        #Nombre de TGVMax
        Nbr_TGVMax = df[df['Prix']=="0"]

        #Verification nombre old
        try:
            old= pd.read_csv("olddata.csv",sep=";")
            oldTGV = old[old['Prix']==0]
            NBR=len(Nbr_TGVMax)-len(oldTGV)
            print(NBR)


        except:
            oldTGV=0    

        df.to_csv("olddata.csv", sep=';')

        if refresh==0:
            return dbc.Table.from_dataframe(df)
        if refresh==1:
            if NBR !=0:
                return (len(Nbr_TGVMax),True)
    





#Main
da = pd.DataFrame()

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Maxime Jenni", href="#")),
        dbc.DropdownMenu(
            nav=True,
            in_navbar=True,
            label="Menu",
            children=[
                dbc.DropdownMenuItem("Login"),
                dbc.DropdownMenuItem("Reset"),
                dbc.DropdownMenuItem(divider=True),
                dbc.DropdownMenuItem("Quitter"),
            ],
        ),
    ],
    brand="TGVMax Booking system",
    brand_href="#",
    sticky="top",
    color="warning"
    
)

body = dbc.Container(
    [
        dbc.Row(
            [
        
                dbc.Col(
                    [
                        html.Br(), #Permet un espace
                                html.H3("Utilisation"),
                                html.P(
                                    "Choisir un point de départ, d'arrivé et la date exacte, puis envoyer les données à Selenium.",
                                    className="card-text",
                                ),
                        html.H3("Départ"),
                        
                        dbc.Checklist(
                            options=[
                                    {"label": "Paris", "value": "Paris"},
                                    {"label": "Strasbourg", "value": "Strasbourg"},
                                    {"label": "Rennes", "value": "Rennes"},
                                    {"label": "Mulhouse", "value": "Mulhouse"},
                            ],
                            value=[],
                            id="checklist-input",
                            ),
                        html.Br(),
                        html.H3("Arrivé"),

                        dbc.Checklist(
                            options=[
                                    {"label": "Paris", "value": "Paris"},
                                    {"label": "Strasbourg", "value": "Strasbourg"},
                                    {"label": "Rennes", "value": "Rennes"},
                                    {"label": "Mulhouse", "value": "Mulhouse"},
                            ],
                            value=[],
                            id="checklist-input2",
                            ),
                    
                        html.Br(),
                        html.H3("Date"),
                        dbc.Input(id="input3", placeholder="JJ/MM/AAAA", type="text"),
                        html.P(id="output3"),
                    
                        dbc.Button("Envoyer les données", id="ok", color="warning",size="md"),
                        html.Br(),
                       
                        
                        
                        ]
                ),

                dbc.Col([
                        html.Br(),
                        html.H3("Tableau"),
                        dbc.Progress(value=0, id='outputall',color="primary",striped=True,animated=True, style={"height": "20px"}),
                        html.Br(),
                        dbc.Table.from_dataframe(df=da, id="tab1",striped=True, bordered=True, hover=True, responsive = True, size="lg")
                ]),

                dbc.Col([
                        html.Br(),
                        html.H3("Auto-Booking"),
                        dbc.Label("Temps de rafraichissement", html_for="slider"),
                        dcc.Interval(id='interval-component', interval=1000,n_intervals=0),
                        dcc.Slider(id="slider", min=0, max=30, step=1, value=5),
                        html.Div(id='slider-output-container'),
                        html.Br(),
                        dbc.Label("Notification", html_for="slider"),
                        dbc.Checklist(
                            options=[
                                    {"label": "SMS", "value": "SMS"},
                                    {"label": "Email", "value": "Email"},
                            ],
                            value=[],
                            id="checklist-input4",
                            ),
                        html.Br(),
                        dbc.Button("Refresh ON/OFF", id="stop", color="danger",size="md"),
                        html.H1(dbc.Button(
                            ["Nombre de TGVMax disponible: ", dbc.Badge(0, id="alert1", color="light")],color="primary")),
                        dbc.Alert("Un nouveau TGVMax est disponible!", id="alert2", color="danger",is_open=False)
                        ])
                
            ]

        )
        
    ]
)



app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = 'TGVMax Booking'
app.layout = html.Div([navbar, body])

@app.callback(
    [Output("tab1", "children"),Output("outputall", "striped")], [Input("ok", "n_clicks")],
    #State permet de ne pas update a chaque changement de valeur, mais uniquement en appuyant sur le bouton ok
    state=[State(component_id='checklist-input', component_property='value'),State(component_id='checklist-input2', component_property='value'),State(component_id='input3', component_property='value')]
    )

def update_output_div(input_ok,checklist_value, checklist_value2, input3_value):

    if input_ok:
        if len(checklist_value)==1 and len(checklist_value2)==1 and input3_value!=None:
            return (myselenium(depart1=checklist_value, arrive1=checklist_value2 ,date1=input3_value,refresh=0,timer=1),True)
    else:
        return ("No data.",False)




@app.callback(
    Output("outputall", "value"), [Input("ok", "n_clicks"),Input('interval-component', 'n_intervals'),Input("tab1", "children")],
    state=[State(component_id='checklist-input', component_property='value'),State(component_id='checklist-input2', component_property='value'),State(component_id='input3', component_property='value')]
    )

def update_output_div2(input_ok,n,tab,checklist_value,checklist_value2, input3_value):
   
    if input_ok:
        if len(checklist_value)==1 and len(checklist_value2)==1 and input3_value!=None:
            return n*2

    elif tab!="No data.":
        return 100
    
    else:
        return 0



@app.callback(
    Output("outputall", "children"),[Input("outputall", "value")]
    )

def update_bartext(input_bar):
    if input_bar<=10:
        return ("Start")
    if 10<input_bar<40:
        return ("Recherche")
    if 40<input_bar<99:
        return ("Chargement en cours")
    if input_bar>=100:
        return ("Incoming")



@app.callback(
    Output('slider-output-container', 'children'),
    [Input('slider', 'value')])

def update_output(value):
    return 'Update toutes les {} minutes.'.format(value)


#Reset boutton
@app.callback(
    Output('interval-component', 'n_intervals'),[Input("ok", "n_clicks")])
    
def update_interval(ok):
    if ok:
        return 0
        

#On/off boutton
@app.callback(
    Output('stop', 'color'),[Input("stop", "n_clicks"),Input('slider', 'value')])


def update_onoff(onoff,slider):
    if onoff % 2 !=0:
        if slider:
            return ("success")
    else :
        return ("danger")


#Nbr de TGV Max alert
@app.callback(
    [Output('alert1', 'children'),Output('alert2', 'is_open')],[Input("stop", "n_clicks"),Input('slider', 'value')],
    state=[State(component_id='checklist-input', component_property='value'),State(component_id='checklist-input2', component_property='value'),State(component_id='input3', component_property='value')]
)

def update_tgv(fresh,slider,checklist_value, checklist_value2, input3_value):
    if fresh % 2 !=0:
        if slider:
            return myselenium(depart1=checklist_value, arrive1=checklist_value2 ,date1=input3_value,refresh=1,timer=slider)
    else :
        return 0,0

app.run_server(host='127.0.0.1', port='8050', debug=False)


