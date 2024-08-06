import uvicorn
from fastapi import FastAPI
import joblib
from api_verif import NBAPlayer

#Permet de lancer l'API REST 
app = FastAPI()
joblib_in = open("SVC_model.joblib","rb")
model=joblib.load(joblib_in)


@app.get('/')
def index():
    return {'message': 'Best future NBA Player ML API'}

@app.post('/predict')
def predict_NBA_player(data:NBAPlayer):
    data = data.dict()
    Game_Played=data['Game_Played']
    Minutes_Played=data['Minutes_Played']
    Points=data['Points']

    prediction = model.predict([[Game_Played, Minutes_Played,Points]])
    proba= model.predict_proba([[Game_Played, Minutes_Played,Points]])
    
    return {
        'prediction': prediction[0],
        'proba': proba[0][1]
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)