from fastapi import FastAPI
from keras.models import load_model
import numpy as np
from pydantic import BaseModel

model = load_model("app/model/wine_model.h5")

app = FastAPI()

class WineFeatures(BaseModel):
    fixed_acidity : float
    volatile_acidity: float
    citric_acid : float
    residual_sugar : float
    chlorides : float
    free_sulfur_dioxide : float
    total_sulfur_dioxide : float
    density : float
    pH: float
    sulphates: float
    alcohol: float
    
@app.get("/")
async def root():
    return {'message': "Wine Prediction API is running"}

@app.post("/predict/")
async def predict(features: WineFeatures):
    feature_array = np.array([[
        features.fixed_acidity, features.volatile_acidity, features.citric_acid,
        features.residual_sugar, features.chlorides, features.free_sulfur_dioxide,
        features.total_sulfur_dioxide, features.density, features.pH,
        features.sulphates, features.alcohol
    ]])
    prediction = model.predict(feature_array)
    wine_type = "Red Wine" if prediction[0][0] > 0.5 else "White Wine"
    return {"prediction": wine_type, "confidence": float(prediction[0][0])}