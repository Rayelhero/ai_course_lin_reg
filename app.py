from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import joblib
import statsmodels.api as sm

app = FastAPI()

p = Path("results.pkl")
print(f"Suche Modell in: {p.absolute()}")

if not p.exists():
    raise FileNotFoundError(f"Model file not found: {p.absolute()}")

model = joblib.load(p)

class Features(BaseModel):
    TESTRESULT: float

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: Features):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = sm.add_constant([[data.TESTRESULT]], has_constant="add")
    prediction = model.predict(X)[0]
    return {
        "prediction": round(float(prediction), 3)
    }
