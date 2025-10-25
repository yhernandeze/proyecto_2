from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd

app = FastAPI(title="Mock Data API", version="1.0")

# Simulamos N batches de 15k filas. Ajusta N según quieras.
TOTAL_BATCHES = 5
CURRENT = {"batch": 0}

COLS = [
    'Elevation','Aspect','Slope',
    'Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points','Wilderness_Area','Soil_Type','Cover_Type'
]

def make_batch_df(n=15000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'Elevation': rng.integers(1800, 3400, n),
        'Aspect': rng.integers(0, 360, n),
        'Slope': rng.integers(0, 45, n),
        'Horizontal_Distance_To_Hydrology': rng.integers(0, 4000, n),
        'Vertical_Distance_To_Hydrology': rng.integers(-500, 500, n),
        'Horizontal_Distance_To_Roadways': rng.integers(0, 7000, n),
        'Hillshade_9am': rng.integers(0, 256, n),
        'Hillshade_Noon': rng.integers(0, 256, n),
        'Hillshade_3pm': rng.integers(0, 256, n),
        'Horizontal_Distance_To_Fire_Points': rng.integers(0, 8000, n),
        'Wilderness_Area': rng.integers(0, 4, n),   # 4 áreas codificadas
        'Soil_Type': rng.integers(0, 40, n),        # 40 tipos suelo
        'Cover_Type': rng.integers(1, 8, n),        # 1..7
    })
    return df

@app.get("/health")
def health():
    return {"status":"ok","batches_total": TOTAL_BATCHES, "next_batch": CURRENT["batch"]+1}

@app.get("/data")
def get_data(group_number: int = 6):
    # Ignoramos el grupo y servimos secuencialmente
    if CURRENT["batch"] >= TOTAL_BATCHES:
        return {"batch_number": CURRENT["batch"], "data": []}  # <- sin datos => DAG marca done
    CURRENT["batch"] += 1
    df = make_batch_df(n=15000, seed=1234 + CURRENT["batch"])
    return {
        "batch_number": CURRENT["batch"],
        "data": df.to_dict(orient="records")
    }
 