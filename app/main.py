from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from src.artifacts import load_artifact

app = FastAPI(title="EPL Win Prediction API")

# Load model once at startup
model, threshold, features = load_artifact()


class MatchFeatures(BaseModel):
    opp_code: int
    gf_rolling: float
    ga_rolling: float
    day_code: int
    venue_code: int


@app.get("/")
def root():
    return {"message": "EPL Prediction API is running."}


@app.post("/predict")
def predict(match: MatchFeatures):
    input_dict = match.dict()

    # Ensure correct feature order
    if features:
        X = pd.DataFrame([[input_dict[f] for f in features]], columns=features)
    else:
        X = pd.DataFrame([input_dict])

    prob = model.predict_proba(X)[0, 1]
    prediction = int(prob >= threshold)

    return {
        "win_probability": round(float(prob), 4),
        "threshold": threshold,
        "prediction": prediction
    }