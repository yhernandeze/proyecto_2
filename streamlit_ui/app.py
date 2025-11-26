import os
import json
from datetime import datetime

import requests
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
BACKEND_URL = os.getenv("INFERENCE_API_URL", "http://inference_api:8989")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "house_prices_regression")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

st.set_page_config(
    page_title="House Prices UI",
    layout="centered",
)

st.title("🏠 House Prices – Demo MLOps (Group 6)")
st.caption(f"Backend: {BACKEND_URL} | Model: `{MODEL_NAME}` stage `{MODEL_STAGE}`")

# -----------------------------------------------------------------------------
# Helper para llamar al API
# -----------------------------------------------------------------------------
def call_predict_api(payload: dict):
    url = f"{BACKEND_URL}/predict"
    try:
        resp = requests.post(url, json=payload, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return data
    except Exception as e:
        st.error(f"Error llamando a {url}: {e}")
        return None


def call_health():
    url = f"{BACKEND_URL}/health"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Error llamando a {url}: {e}")
        return None


# -----------------------------------------------------------------------------
# Estado de sesión para histórico local (UI)
# -----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []  # lista de dicts {timestamp, features, prediction}


# -----------------------------------------------------------------------------
# Layout en tabs
# -----------------------------------------------------------------------------
tab_predict, tab_history, tab_info = st.tabs(["🔮 Predicción", "📜 Historial (UI)", "ℹ️ Info modelo"])

# -----------------------------------------------------------------------------
# TAB 1: Predicción
# -----------------------------------------------------------------------------
with tab_predict:
    st.subheader("Parámetros de la casa")

    col1, col2 = st.columns(2)

    with col1:
        bed = st.number_input("Dormitorios (bed)", min_value=0.0, max_value=20.0, value=3.0, step=1.0)
        bath = st.number_input("Baños (bath)", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
        acre_lot = st.number_input("Tamaño del lote (acre_lot)", min_value=0.0, max_value=10.0, value=0.3, step=0.01)
        house_size = st.number_input("Tamaño de la casa (house_size)", min_value=0.0, max_value=20000.0, value=1200.0, step=10.0)

    with col2:
        zip_code = st.number_input("Zip code", min_value=0.0, max_value=99999.0, value=1001.0, step=1.0)
        brokered_by = st.number_input("Brokered by (id)", min_value=0.0, max_value=200000.0, value=67455.0, step=1.0)
        street = st.number_input("Street (id)", min_value=0.0, max_value=2000000.0, value=1698080.0, step=10.0)

    if st.button("Predecir precio"):
        features = {
            "bed": bed,
            "bath": bath,
            "acre_lot": acre_lot,
            "house_size": house_size,
            "zip_code": zip_code,
            "brokered_by": brokered_by,
            "street": street,
        }

        st.write("Payload enviado al API:")
        st.json(features)

        result = call_predict_api(features)
        if result is not None and "predicted_price" in result:
            pred = result["predicted_price"]
            st.success(f"Precio predicho: **${pred:,.2f}**")

            # Guardar en historial local
            st.session_state["history"].append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "features": features,
                    "prediction": pred,
                }
            )

# -----------------------------------------------------------------------------
# TAB 2: Historial (local en UI)
# -----------------------------------------------------------------------------
with tab_history:
    st.subheader("Historial de predicciones (solo sesión actual)")

    if not st.session_state["history"]:
        st.info("Aún no hay predicciones en esta sesión.")
    else:
        # Convertir a DataFrame
        rows = []
        for item in st.session_state["history"]:
            row = {
                "timestamp": item["timestamp"],
                **item["features"],
                "predicted_price": item["prediction"],
            }
            rows.append(row)
        df_hist = pd.DataFrame(rows)
        st.dataframe(df_hist)

# -----------------------------------------------------------------------------
# TAB 3: Info modelo / health
# -----------------------------------------------------------------------------
with tab_info:
    st.subheader("Estado del API y del modelo")

    if st.button("Comprobar /health"):
        health = call_health()
        if health is not None:
            st.success("Respuesta de /health:")
            st.json(health)

    st.markdown("### Configuración actual")
    st.code(
        f"""
INFERENCE_API_URL = {BACKEND_URL}
MLFLOW_TRACKING_URI = {MLFLOW_TRACKING_URI}
MODEL_NAME = {MODEL_NAME}
MODEL_STAGE = {MODEL_STAGE}
""",
        language="bash",
    )

    st.markdown(
        """
En esta versión básica:

- Las peticiones se envían al endpoint `/predict` del API FastAPI.
- El historial mostrado es **solo local de la sesión de Streamlit**.
- Más adelante podemos:
  - Leer el histórico real desde la tabla `house_price_inference_log`.
  - Mostrar métricas agregadas de producción.
  - Añadir pestaña de SHAP / explicabilidad.
"""
    )
