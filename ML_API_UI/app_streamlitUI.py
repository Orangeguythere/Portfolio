import uvicorn
from fastapi import FastAPI
import joblib
from api_verif import NBAPlayer
import requests
import streamlit as st


def add_bg_from_local():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://img.freepik.com/free-photo/basketball-game-concept_23-2150910694.jpg?t=st=1719576996~exp=1719580596~hmac=be3b9825287c6e30d13ebfbc6d630794390d30607beebea9303d05873eeaf0a1&w=1800");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local()


# Define the title
st.title("NBA Players evaluation web application")
st.write(
    "The model evaluates a player level for NBA based on the inputs below."
)

# Input 1
Game_Played = st.number_input("Number of Games played", value=None, placeholder="Type a number...")
st.write("Current number of games ", Game_Played)

# Input 1
Minutes_Played = st.number_input("Average of minutes played", value=None, placeholder="Type a number...")
st.write("Minutes played ", Minutes_Played)

# Input 1
Points = st.number_input("Number of points ", value=None, placeholder="Type a number...")
st.write("Current number of points ", Points)

# Class values to be returned by the model
class_values = {
    0: "not a good choice",
    1: "a good choice"
    }

# When 'Submit' is selected
if st.button("Submit"):

    # Inputs to ML model
    inputs = (
            {
                "Game_Played": Game_Played,
                "Minutes_Played": Minutes_Played,
                "Points": Points
            })


    # Posting inputs to ML API
    response = requests.post(url = 'http://127.0.0.1:8000/predict', json=inputs, verify=False)
    json_response = response.json()

    prediction = class_values[json_response.get("prediction")]
    proba= json_response.get("proba")
    #prediction = json_response.get("prediction")

    st.subheader(f"This player is **{prediction}!**")
    st.subheader(f"Probability is **{round(proba,2)*100}%**")