from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel
from fastapi.responses import FileResponse
import os
import pandas as pd

app = FastAPI(title="NIDS Prediction API")

# Load models
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "Model"

rf_model = joblib.load(MODEL_DIR / "rf_model.pkl")
xgb_model = joblib.load(MODEL_DIR / "xgb_model.pkl")
label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")
# feature_columns = 78 #For testing


class InputData(BaseModel):
    features: list


# Validate feature length
def validate_input(features):
    if len(features) != len(feature_columns):
        raise HTTPException(
            status_code=400,
            detail=f"Expected len(feature_columns) features, got {len(features)}"
        )


@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join("static", "index.html"))


@app.post("/predict/rf")
def predict_rf(data: InputData):

    validate_input(data.features)

    x = pd.DataFrame([data.features], columns=feature_columns)
    pred = rf_model.predict(x)

    return {
        "prediction": label_encoder.inverse_transform(pred)[0]
    }


@app.post("/predict/xgb")
def predict_xgb(data: InputData):

    validate_input(data.features)

    x = np.array(data.features).reshape(1, -1)
    pred = xgb_model.predict(x)

    return {
        "prediction": label_encoder.inverse_transform(pred)[0]
    }