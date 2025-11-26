import os
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import mlflow
import mlflow.pyfunc

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "house_prices_regression")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

app = FastAPI(title="House Prices Inference API")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


class HouseFeatures(BaseModel):
    bed: float
    bath: float
    acre_lot: float
    house_size: float
    zip_code: float
    brokered_by: float
    street: float


class PredictionResponse(BaseModel):
    predicted_price: float


@lru_cache(maxsize=1)
def load_model():
    """
    Load the model from MLflow Model Registry.
    """
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from '{model_uri}': {e}")


@app.get("/health")
def health():
    """
    Simple health endpoint.
    """
    try:
        _ = load_model()
        return {"status": "ok", "model_name": MODEL_NAME, "stage": MODEL_STAGE}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    """
    Predict house price given feature set.
    """
    try:
        model = load_model()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Convert to DataFrame with same feature order as training
    df = pd.DataFrame(
        [{
            "bed": features.bed,
            "bath": features.bath,
            "acre_lot": features.acre_lot,
            "house_size": features.house_size,
            "zip_code": features.zip_code,
            "brokered_by": features.brokered_by,
            "street": features.street,
        }]
    )

    try:
        preds = model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    return PredictionResponse(predicted_price=float(preds[0]))
