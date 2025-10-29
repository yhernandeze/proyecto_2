from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import os
import logging
from datetime import datetime
import traceback

import numpy as np
import pandas as pd

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.pyfunc import load_model
from mlflow.models import get_model_info

# ---------- logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- MLflow config ----------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME")        # e.g. "diabetes_readmit_rf"
DEFAULT_MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

# ---------- FastAPI ----------
app = FastAPI(
    title="Generic MLflow Inference API",
    description="Loads an MLflow (pyfunc) registered model and serves predict()",
    version="1.1.0"
)

# In-memory cache
_loaded = {
    "model_name": None,
    "stage": None,
    "version": None,
    "model": None,
    "signature": None,
    "loaded_at": None
}

# ---------- helpers ----------
def _model_uri(name: str, version: Optional[str], stage: Optional[str]) -> str:
    if version:
        return f"models:/{name}/{version}"
    if stage:
        return f"models:/{name}/{stage}"
    return f"models:/{name}/latest"

def _load(name: str, version: Optional[str] = None, stage: Optional[str] = None):
    uri = _model_uri(name, version, stage)
    logger.info(f"Loading MLflow model from URI: {uri}")
    model = load_model(uri)  # pyfunc model
    info = get_model_info(uri)
    _loaded.update({
        "model_name": name,
        "stage": stage,
        "version": version,
        "model": model,
        "signature": info.signature.dict() if info and info.signature else None,
        "loaded_at": datetime.now().isoformat()
    })
    logger.info(f"Loaded model '{name}' (stage={stage}, version={version})")
    return model

def _ensure_loaded():
    if _loaded["model"] is None:
        if not DEFAULT_MODEL_NAME:
            raise HTTPException(status_code=503, detail="No default MODEL_NAME configured")
        try:
            _load(DEFAULT_MODEL_NAME, stage=DEFAULT_MODEL_STAGE)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"No se pudo cargar el modelo por defecto: {e}")

def _extract_sig_columns(sig_dict: Optional[dict]) -> Optional[List[str]]:
    """If the MLflow signature exists, return ordered input column names."""
    if not sig_dict:
        return None
    # MLflow pydantic signature dict → fields under inputs
    try:
        inputs = sig_dict.get("inputs")
        if not inputs:
            return None
        # support schema stored as a list of dicts with 'name' keys
        names = [f.get("name") for f in inputs if isinstance(f, dict) and "name" in f]
        return names if names else None
    except Exception:
        return None

# ---------- default-fill config for Diabetes dataset ----------
NUM_DEFAULT_0 = {
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    "admission_type_id", "discharge_disposition_id", "admission_source_id"
}
CAT_DEFAULT_UNKNOWN = {
    "race", "gender", "age", "payer_code", "medical_specialty",
    "max_glu_serum", "A1Cresult"
}
MED_COLS = {
    "metformin","repaglinide","nateglinide","chlorpropamide","glimepiride",
    "acetohexamide","glipizide","glyburide","tolbutamide","pioglitazone",
    "rosiglitazone","acarbose","miglitol","troglitazone","tolazamide",
    "examide","citoglipton","glyburide-metformin","glipizide-metformin",
    "glimepiride-pioglitazone","metformin-rosiglitazone","metformin-pioglitazone",
    "insulin","change","diabetesMed"
}
FILL_VALUES = {
    **{c: 0 for c in NUM_DEFAULT_0},
    **{c: "Unknown" for c in CAT_DEFAULT_UNKNOWN},
    **{c: "No" for c in MED_COLS},
    "weight": "?",  # dataset used "?" for missing weight
}

def _try_predict(df: pd.DataFrame):
    """Call underlying model.predict, return (ok, result_or_error)."""
    try:
        res = _loaded["model"].predict(df)
        return True, res
    except Exception as e:
        return False, e

def _missing_from_exception(err: Exception) -> List[str]:
    """Parse KeyError like: \"['col1','col2'] not in index\" to list of col names."""
    msg = str(err)
    if "not in index" in msg:
        start = msg.find("[")
        end = msg.find("]", start)
        if start != -1 and end != -1:
            inside = msg[start+1:end]
            parts = [p.strip().strip("'").strip('"') for p in inside.split(",")]
            return [p for p in parts if p]
    return []

# ---------- schemas ----------
class PredictBody(BaseModel):
    records: List[Dict[str, Any]] = Field(..., description="List of feature dictionaries")

class LoadBody(BaseModel):
    model_name: str
    version: Optional[str] = None
    stage: Optional[str] = None

# ---------- endpoints ----------
@app.get("/")
def root():
    return {
        "message": "MLflow Inference API (pyfunc)",
        "status": "running",
        "mlflow_uri": MLFLOW_TRACKING_URI,
        "default_model_name": DEFAULT_MODEL_NAME,
        "default_model_stage": DEFAULT_MODEL_STAGE
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "loaded": _loaded["model"] is not None
    }

@app.get("/models")
def list_models():
    try:
        client = MlflowClient()
        rms = client.search_registered_models()
        return [rm.name for rm in rms]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/load")
def load_specific_model(body: LoadBody):
    try:
        _load(body.model_name, version=body.version, stage=body.stage)
        return {
            "message": "Modelo cargado",
            "model_name": _loaded["model_name"],
            "stage": _loaded["stage"],
            "version": _loaded["version"],
            "loaded_at": _loaded["loaded_at"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al cargar modelo: {e}")

@app.get("/model/expected-schema")
def expected_schema():
    """Show the input schema according to MLflow model signature (if logged)."""
    _ensure_loaded()
    sig = _loaded["signature"]
    return {
        "model_name": _loaded["model_name"],
        "stage": _loaded["stage"],
        "version": _loaded["version"],
        "signature": sig
    }

@app.post("/predict/check")
def predict_check(body: PredictBody):
    """Dry-run: return which columns are missing (no 500)."""
    _ensure_loaded()
    if not body.records:
        raise HTTPException(status_code=400, detail="Empty records")
    df = pd.DataFrame(body.records)

    ok, res = _try_predict(df)
    if ok:
        return {"missing_columns": [], "note": "predict would succeed"}
    miss = _missing_from_exception(res)
    return {"missing_columns": miss, "raw_error": str(res)}

@app.post("/predict")
def predict(body: PredictBody):
    """
    Predict with arbitrary JSON records.
    If signature exists: align to signature.
    If not: try once, and on KeyError (missing columns) auto-fill with defaults & retry once.
    """
    _ensure_loaded()
    if not body.records:
        raise HTTPException(status_code=400, detail="Empty records")

    df = pd.DataFrame(body.records)
    logger.info(f"Incoming DF columns: {list(df.columns)} | shape={df.shape}")

    sig_cols = _extract_sig_columns(_loaded["signature"])
    if sig_cols:
        # fill missing signature columns with NaN and reorder
        for c in sig_cols:
            if c not in df.columns:
                df[c] = np.nan
        df = df[sig_cols]
        logger.info(f"Aligned to signature. Final DF columns: {list(df.columns)}")

    ok, res = _try_predict(df)
    if ok:
        preds = res
    else:
        # only retry on missing-column errors
        missing_cols = _missing_from_exception(res)
        if not missing_cols:
            logger.error("Prediction failed (no missing-column hint). Stack:\n" + traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error en predicción: {res}")

        logger.warning(f"Missing columns detected: {missing_cols}. Auto-filling once and retrying.")
        for c in missing_cols:
            df[c] = FILL_VALUES.get(c, np.nan)

        if sig_cols:
            # ensure all signature columns exist & reorder again
            for c in sig_cols:
                if c not in df.columns:
                    df[c] = np.nan
            df = df[sig_cols]

        ok2, res2 = _try_predict(df)
        if not ok2:
            logger.error("Prediction failed after auto-fill retry. Stack:\n" + traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error en predicción (tras reintento): {res2}")
        preds = res2

    preds_out = preds.tolist() if hasattr(preds, "tolist") else preds
    return {
        "model_name": _loaded["model_name"],
        "stage": _loaded["stage"],
        "version": _loaded["version"],
        "timestamp": datetime.now().isoformat(),
        "predictions": preds_out
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8989)
 