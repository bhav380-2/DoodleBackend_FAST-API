from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow import keras
from typing import List
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI()

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

file_path = './categories.json' 
categories = None

with open(file_path, 'r') as file:
    categories = json.load(file) 
# print(categories)
model = keras.models.load_model('CNN1.keras')

class PredictionRequest(BaseModel):
    data: List[int]

@app.get("/")
async def home():
    return {"success": True}

@app.post("/predict")
async def predict(request: PredictionRequest):
    # Extract data from request
    # print(request.data)
    data = np.array(request.data)
    
    image = np.array(data).reshape(1, 64, 64, 1)
    
    # Run the prediction
    y = model.predict(image, verbose=1)
    # print("******************")
    # print(y)
    top_3 = np.argsort(-y)[:, 0:3]  # Get top 3 predictions

    preds = []
    for p in top_3[0].tolist():
        preds.append(categories[str(p)])

    return preds

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=True)
