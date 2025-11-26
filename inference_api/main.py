import os
import time
from functools import lru_cache

import pandas as pd
import pymysql
from fastapi import FastAPI, HTTPException
from fastapi import Response
from pydantic import BaseModel

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# ---------------- ENV VARS ----------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "house_prices_regression")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

# Reutilizamos el mismo URI de MySQL que Airflow, para loguear inferencias
DATA_DB_URI = os.getenv("DATA_DB_URI")  # mysql+pymysql://user:pass@mysql:3306/datasets_db

app = FastAPI(title="House Prices Inference API")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ------------- Prometheus metrics -------------
PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Total prediction requests",
    ["status"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds"
)

# ------------- DB helpers -------------
def _parse_mysql_uri(uri: str):
    if not uri:
        raise ValueError("DATA_DB_URI is not set")
    prefix, rest = uri.split("://", 1)
    creds, hostpart = rest.split("@", 1)
    user, password = creds.split(":", 1)
    hostport, dbname = hostpart.split("/", 1)
    if ":" in hostport:
        host, port = hostport.split(":", 1)
        port = int(port)
    else:
        host, port = hostport, 3306
    return {
        "user": user,
        "password": password,
        "host": host,
        "port": port,
        "database": dbname,
    }

def _get_mysql_connection():
    cfg = _parse_mysql_uri(DATA_DB_URI)
    return pymysql.connect(
        host=cfg["host"],
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
        port=cfg["port"],
        cursorclass=pymysql.cursors.DictCursor,
    )

# ------------- Pydantic models -------------
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
    model_name: str
    model_version: str
    run_id: str

# ------------- Model loading -------------
@lru_cache(maxsize=1)
def load_model():
    """
    Load the model from MLflow Model Registry and cache it.
    Also return model metadata (version, run_id).
    """
    client = MlflowClient()

    # Obtener la última versión en el stage solicitado
    versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
    if not versions:
        raise RuntimeError(
            f"No model versions found for name='{MODEL_NAME}' and stage='{MODEL_STAGE}'"
        )
    v = versions[0]
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from '{model_uri}': {e}")

    return {
        "model": model,
        "model_name": MODEL_NAME,
        "model_version": v.version,
        "run_id": v.run_id,
    }

def _log_inference_to_db(model_info: dict, features: HouseFeatures, predicted_price: float):
    """
    Insert one row into inference_requests table.
    Errors here should not break the API.
    """
    if not DATA_DB_URI:
        # No DB configured, silenciosamente no logueamos
        print("[inference_log] DATA_DB_URI not set; skipping DB logging.")
        return

    try:
        conn = _get_mysql_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO inference_requests (
                      model_name,
                      model_version,
                      run_id,
                      bed,
                      bath,
                      acre_lot,
                      house_size,
                      zip_code,
                      brokered_by,
                      street,
                      predicted_price
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        model_info["model_name"],
                        str(model_info["model_version"]),
                        model_info["run_id"],
                        float(features.bed),
                        float(features.bath),
                        float(features.acre_lot),
                        float(features.house_size),
                        float(features.zip_code),
                        float(features.brokered_by),
                        float(features.street),
                        float(predicted_price),
                    ),
                )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        # No queremos romper la predicción solo por el logging
        print(f"[inference_log] Failed to log inference to DB: {e}")

# ------------- Endpoints -------------
@app.get("/health")
def health():
    """
    Simple health endpoint that also checks model loading.
    """
    try:
        mi = load_model()
        return {
            "status": "ok",
            "model_name": mi["model_name"],
            "stage": MODEL_STAGE,
            "model_version": mi["model_version"],
            "run_id": mi["run_id"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    """
    Predict house price given feature set.
    """
    start = time.time()
    try:
        mi = load_model()
        model = mi["model"]

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

        preds = model.predict(df)
        predicted_price = float(preds[0])

        # Log to DB (best-effort)
        _log_inference_to_db(mi, features, predicted_price)

        PREDICTION_REQUESTS.labels(status="success").inc()
        return PredictionResponse(
            predicted_price=predicted_price,
            model_name=mi["model_name"],
            model_version=str(mi["model_version"]),
            run_id=mi["run_id"],
        )
    except HTTPException as e:
        PREDICTION_REQUESTS.labels(status=str(e.status_code)).inc()
        raise
    except Exception as e:
        PREDICTION_REQUESTS.labels(status="500").inc()
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    finally:
        elapsed = time.time() - start
        PREDICTION_LATENCY.observe(elapsed)


@app.get("/metrics")
def metrics():
    """
    Prometheus metrics endpoint.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
