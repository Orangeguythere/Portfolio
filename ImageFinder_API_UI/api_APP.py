import uvicorn
import os
from fastapi import FastAPI,File, UploadFile, Request
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import joblib
from Image_findder import find_similar_images,extract_features


#Permet de lancer l'API REST 
app = FastAPI()



@app.post("/find_similar")
async def api_find_similar(file: UploadFile = File(...), n: int = 3):
    contents = await file.read()
    img = Image.open(BytesIO(contents))
    
    input_features = extract_features(img)
    if input_features is None:
        return JSONResponse(content={"error": "Unable to process the uploaded image"}, status_code=400)
    
    similar_images = find_similar_images(input_features, n)
    
    results = [{"id": filename, "similarity": float(similarity)} for filename, similarity in similar_images]
    return JSONResponse(content=results)


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)