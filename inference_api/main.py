from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de MLflow
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Crear aplicación FastAPI
app = FastAPI(
    title="Forest Cover Type Prediction API",
    description="API para inferencia de tipos de cobertura forestal usando MLflow models",
    version="1.0.0"
)

# Variable global para almacenar modelos cargados
loaded_models = {}
current_model_name = None

class PredictionInput(BaseModel):
    """Schema para entrada de predicción"""
    Elevation: float = Field(..., description="Elevación en metros")
    Aspect: float = Field(..., description="Aspecto en grados azimuth")
    Slope: float = Field(..., description="Pendiente en grados")
    Horizontal_Distance_To_Hydrology: float = Field(..., description="Distancia horizontal al agua")
    Vertical_Distance_To_Hydrology: float = Field(..., description="Distancia vertical al agua")
    Horizontal_Distance_To_Roadways: float = Field(..., description="Distancia horizontal a carreteras")
    Hillshade_9am: int = Field(..., ge=0, le=255, description="Índice de sombra a las 9am")
    Hillshade_Noon: int = Field(..., ge=0, le=255, description="Índice de sombra al mediodía")
    Hillshade_3pm: int = Field(..., ge=0, le=255, description="Índice de sombra a las 3pm")
    Horizontal_Distance_To_Fire_Points: float = Field(..., description="Distancia a puntos de fuego")
    
    class Config:
        json_schema_extra = {
            "example": {
                "Elevation": 2596,
                "Aspect": 51,
                "Slope": 3,
                "Horizontal_Distance_To_Hydrology": 258,
                "Vertical_Distance_To_Hydrology": 0,
                "Horizontal_Distance_To_Roadways": 510,
                "Hillshade_9am": 221,
                "Hillshade_Noon": 232,
                "Hillshade_3pm": 148,
                "Horizontal_Distance_To_Fire_Points": 6279
            }
        }

class PredictionOutput(BaseModel):
    """Schema para salida de predicción"""
    prediction: int = Field(..., description="Tipo de cobertura forestal predicho (1-7)")
    prediction_label: str = Field(..., description="Etiqueta del tipo de cobertura")
    confidence: Optional[float] = Field(None, description="Confianza de la predicción")
    model_used: str = Field(..., description="Nombre del modelo utilizado")
    timestamp: str = Field(..., description="Timestamp de la predicción")

class BatchPredictionInput(BaseModel):
    """Schema para predicciones por lote"""
    data: List[PredictionInput]

class ModelInfo(BaseModel):
    """Información del modelo"""
    model_name: str
    version: Optional[str]
    stage: Optional[str]
    loaded: bool

def get_cover_type_label(prediction: int) -> str:
    """Convertir predicción numérica a etiqueta"""
    labels = {
        1: "Spruce/Fir",
        2: "Lodgepole Pine",
        3: "Ponderosa Pine",
        4: "Cottonwood/Willow",
        5: "Aspen",
        6: "Douglas-fir",
        7: "Krummholz"
    }
    return labels.get(prediction, "Unknown")

def load_model_from_mlflow(model_name: str, version: Optional[str] = None, stage: Optional[str] = "Production"):
    """
    Cargar modelo desde MLflow
    """
    global loaded_models, current_model_name
    
    try:
        logger.info(f"Intentando cargar modelo: {model_name}")
        
        # Si se especifica versión
        if version:
            model_uri = f"models:/{model_name}/{version}"
        # Si se especifica stage
        elif stage:
            model_uri = f"models:/{model_name}/{stage}"
        else:
            # Cargar última versión
            model_uri = f"models:/{model_name}/latest"
        
        logger.info(f"URI del modelo: {model_uri}")
        
        # Cargar modelo
        model = mlflow.sklearn.load_model(model_uri)
        
        # Guardar en cache
        loaded_models[model_name] = {
            'model': model,
            'version': version,
            'stage': stage,
            'loaded_at': datetime.now().isoformat()
        }
        
        current_model_name = model_name
        logger.info(f"Modelo {model_name} cargado exitosamente")
        
        return model
    
    except Exception as e:
        logger.error(f"Error al cargar modelo {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al cargar modelo: {str(e)}")

@app.get("/")
def root():
    """Endpoint raíz"""
    return {
        "message": "Forest Cover Type Prediction API",
        "version": "1.0.0",
        "status": "running",
        "mlflow_uri": MLFLOW_TRACKING_URI
    }

@app.get("/health")
def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "mlflow_connection": MLFLOW_TRACKING_URI
    }

@app.get("/models", response_model=List[str])
def list_available_models():
    """
    Listar modelos disponibles en MLflow
    """
    try:
        client = mlflow.tracking.MlflowClient()
        registered_models = client.search_registered_models()
        
        model_names = [rm.name for rm in registered_models]
        return model_names
    
    except Exception as e:
        logger.error(f"Error al listar modelos: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al listar modelos: {str(e)}")

@app.get("/models/{model_name}/info", response_model=ModelInfo)
def get_model_info(model_name: str):
    """
    Obtener información de un modelo específico
    """
    is_loaded = model_name in loaded_models
    
    if is_loaded:
        model_data = loaded_models[model_name]
        return ModelInfo(
            model_name=model_name,
            version=model_data.get('version'),
            stage=model_data.get('stage'),
            loaded=True
        )
    else:
        return ModelInfo(
            model_name=model_name,
            version=None,
            stage=None,
            loaded=False
        )

@app.post("/models/{model_name}/load")
def load_specific_model(
    model_name: str,
    version: Optional[str] = None,
    stage: Optional[str] = "Production"
):
    """
    BONO: Cargar un modelo específico para usar en las predicciones
    """
    try:
        load_model_from_mlflow(model_name, version, stage)
        return {
            "message": f"Modelo {model_name} cargado exitosamente",
            "model_name": model_name,
            "version": version,
            "stage": stage,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    """
    Realizar predicción individual
    """
    global current_model_name
    
    # Si no hay modelo cargado, intentar cargar uno por defecto
    if not current_model_name or current_model_name not in loaded_models:
        try:
            # Intentar cargar modelo por defecto
            available_models = list_available_models()
            if not available_models:
                raise HTTPException(status_code=503, detail="No hay modelos disponibles en MLflow")
            
            # Cargar el primer modelo disponible que contenga "forest_cover"
            default_model = next((m for m in available_models if 'forest_cover' in m.lower()), available_models[0])
            load_model_from_mlflow(default_model)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"No se pudo cargar ningún modelo: {str(e)}")
    
    try:
        # Preparar datos para predicción
        input_dict = input_data.model_dump()
        df = pd.DataFrame([input_dict])
        
        # Realizar predicción
        model_data = loaded_models[current_model_name]
        model = model_data['model']
        
        prediction = int(model.predict(df)[0])
        
        # Obtener probabilidades si el modelo lo soporta
        confidence = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(df)[0]
            confidence = float(max(proba))
        
        return PredictionOutput(
            prediction=prediction,
            prediction_label=get_cover_type_label(prediction),
            confidence=confidence,
            model_used=current_model_name,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.post("/predict/batch")
def predict_batch(batch_input: BatchPredictionInput):
    """
    Realizar predicciones por lote
    """
    global current_model_name
    
    if not current_model_name or current_model_name not in loaded_models:
        raise HTTPException(status_code=503, detail="No hay modelo cargado. Use /models/{model_name}/load primero")
    
    try:
        # Preparar datos
        input_list = [item.model_dump() for item in batch_input.data]
        df = pd.DataFrame(input_list)
        
        # Realizar predicciones
        model_data = loaded_models[current_model_name]
        model = model_data['model']
        
        predictions = model.predict(df).tolist()
        
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "index": i,
                "prediction": int(pred),
                "prediction_label": get_cover_type_label(int(pred))
            })
        
        return {
            "predictions": results,
            "model_used": current_model_name,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error en predicción batch: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en predicción batch: {str(e)}")

@app.get("/current-model")
def get_current_model():
    """
    Obtener información del modelo actualmente en uso
    """
    if not current_model_name or current_model_name not in loaded_models:
        return {"message": "No hay modelo cargado actualmente"}
    
    model_data = loaded_models[current_model_name]
    return {
        "model_name": current_model_name,
        "version": model_data.get('version'),
        "stage": model_data.get('stage'),
        "loaded_at": model_data.get('loaded_at')
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8989)