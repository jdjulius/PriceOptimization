from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

class PredictionInput(BaseModel):
    Store_ID: str
    Item_ID: str
    Price: float
    Item_Quantity: int
    Competition_Price: float

app = FastAPI(title="Price Optimization API")

# Load model and preprocessor on startup
model_bundle = joblib.load("model.pkl")
model = model_bundle["model"]
preprocessor = model_bundle["preprocessor"]

@app.post("/predict")
def predict(data: PredictionInput):
    df = pd.DataFrame([data.dict()])
    processed = preprocessor.transform(df)
    prediction = model.predict(processed)[0]
    return {"prediction": prediction}
