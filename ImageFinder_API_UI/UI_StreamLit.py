import uvicorn
from fastapi import FastAPI
import joblib
import requests
import streamlit as st


def add_bg_from_local():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://mrwallpaper.com/images/hd/80s-retro-neon-background-qik4m1gn00savcwo.jpg");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

#add_bg_from_local()



st.title("ImageFinder")
st.write("The model evaluates a picture similarity with a local database.")

# Input for number of similar images
n = st.number_input("Number of similar images", min_value=1, max_value=10, value=3, step=1)
st.write(f"Current number of images: {n}")

# Image Upload
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Find Similar Images"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://localhost:8000/find_similar", files=files, params={"n": n})
        
        if response.status_code == 200:
            results = response.json()
            st.subheader("Similar Images:")
            for result in results:
                st.write(f"Image ID: {result['id']}")
                st.subheader(f"Similarity: {result['similarity']:.4f}")

        else:
            st.error("Error processing the image. Please try again.")


