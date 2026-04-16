
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import statsmodels.api as sm

app = FastAPI()
from pathlib import Path
p = Path("14_N8N/results.pkl")  # Voll relativ zu /app
print(f"Suche Modell in: {p.absolute()}")  # Debug-Log
if p.exists():
    model
    
class Features(BaseModel):
    TESTRESULT: float

@app.post("/predict")
def predict(data: Features):
    X = sm.add_constant([[data.TESTRESULT]], has_constant='add')
    prediction = model.predict(X)[0]
    return {
        "prediction": round(float(prediction), 3)
    }
