import os
from datetime import datetime

import mlflow
import pandas as pd
import requests
import shap
import streamlit as st

from shared.feature_contract import FEATURE_COLUMNS, get_feature_names as shared_get_feature_names

# CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="Predicción de precios de vivienda",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Colores corporativos
PRIMARY = "#004A9F"   # azul
ACCENT = "#00A19B"    # teal
BG_LIGHT = "#F5F7FB"  # gris muy claro
CARD_BG = "#FFFFFF"   # blanco

# Endpoint de la API de inferencia (desde variable de entorno)
INFERENCE_API_URL = os.getenv("INFERENCE_API_URL", "http://localhost:8989")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "house_prices_regression")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

# ESTILOS
st.markdown(
    f"""
    <style>
    .main {{
        background-color: {BG_LIGHT};
    }}

    .big-title {{
        font-size: 30px;
        font-weight: 700;
        color: {PRIMARY};
        margin-bottom: 0.2rem;
    }}

    .subtitle {{
        font-size: 14px;
        color: #4F4F4F;
        margin-bottom: 1.5rem;
    }}

    .card {{
        background-color: {CARD_BG};
        padding: 1.2rem 1.0rem;
        border-radius: 10px;
        box-shadow: 0 0 8px rgba(15, 23, 42, 0.06);
        border-left: 4px solid {ACCENT};
        margin-bottom: 0.8rem;
    }}

    .section-title {{
        font-size: 18px;
        font-weight: 600;
        color: {PRIMARY};
        margin-top: 1.8rem;
        margin-bottom: 0.5rem;
    }}

    .small-label {{
        font-size: 12px;
        color: #6B7280;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# FUNCIONES AUXILIARES
DEFAULT_BASELINE = pd.DataFrame([{
    "bed": 3,
    "bath": 2,
    "acre_lot": 0.1,
    "house_size": 1200,
    "zip_code": 12345,
    "brokered_by": 60000,
    "street": 1000,
}])

def build_feature_df(payload: dict) -> pd.DataFrame:
    """Create DataFrame with proper column order for the model."""
    return pd.DataFrame([{col: payload[col] for col in FEATURE_COLUMNS}])

def call_inference_api(payload: dict):
    """
    Llama a la API de inferencia con los datos de la vivienda.
    Se asume un endpoint POST /predict que devuelve un JSON con
    la predicción, por ejemplo:
        { "prediction": 123456.78 }
    o
        { "predicted_price": 123456.78 }
    """
    try:
        url = INFERENCE_API_URL.rstrip("/") + "/predict"
        resp = requests.post(url, json=payload, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        # Intentamos leer distintas claves posibles
        if isinstance(data, dict):
            if "prediction" in data:
                return float(data["prediction"]), data
            if "predicted_price" in data:
                return float(data["predicted_price"]), data

        # Si el formato es distinto, devolvemos el JSON completo
        return None, data

    except Exception as e:
        return None, {"error": str(e)}


@st.cache_resource(show_spinner=False)
def load_model_and_explainer():
    """Load the MLflow model and prepare a SHAP explainer (cached)."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.pyfunc.load_model(model_uri)
    # Use a small background sample to keep SHAP cheap
    explainer = shap.Explainer(model.predict, DEFAULT_BASELINE)
    return model, explainer


@st.cache_data(show_spinner=False)
def compute_shap_values(explainer, df: pd.DataFrame):
    """Compute SHAP values for a single-row dataframe."""
    explanation = explainer(df)
    shap_values = explanation.values[0]
    base_value = explanation.base_values[0]
    return shap_values, base_value


# SIDEBAR – INFO GENERAL
st.sidebar.title("🏠 Predicción de precio")

st.sidebar.markdown(
    """
Esta aplicación permite **estimar el precio de una vivienda**
a partir de sus características principales:

- Habitaciones (`bed`)
- Baños (`bath`)
- Tamaño del lote (`acre_lot`)
- Tamaño construido (`house_size`)
- Código postal (`zip_code`)
- Agencia o corredor (`brokered_by`)
- Calle (`street`)
"""
)

st.sidebar.markdown("---")
st.sidebar.caption("Fuente del modelo:")
st.sidebar.write("Modelo de regresión entrenado en el pipeline MLOps\n(MLflow + Inference API).")

st.sidebar.markdown("---")
st.sidebar.caption("Última actualización de la UI:")
st.sidebar.write(datetime.now().strftime("%Y-%m-%d %H:%M"))


# CABECERA
st.markdown('<div class="big-title">Estimador de precio de vivienda</div>', unsafe_allow_html=True)
st.markdown(
    """
<div class="subtitle">
Introduce las características de la vivienda y obtén una estimación del precio
según el modelo entrenado en el proyecto MLOps.
</div>
""",
    unsafe_allow_html=True,
)

# LAYOUT PRINCIPAL: FORMULARIO (IZQ) + RESULTADO (DER)
col_form, col_result = st.columns([1.2, 1])

with col_form:
    st.markdown('<div class="section-title">📋 Datos de la vivienda</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # --- Inputs numéricos básicos ---
    bed = st.number_input("Habitaciones (bed)", min_value=0, max_value=20, value=3, step=1)
    bath = st.number_input("Baños (bath)", min_value=0.0, max_value=20.0, value=2.0, step=0.5)
    acre_lot = st.number_input("Tamaño del lote (acre_lot)", min_value=0.0, value=0.1, step=0.1)
    house_size = st.number_input("Tamaño construido (house_size, ft²)", min_value=0, value=1200, step=10)
    zip_code = st.number_input("Código postal (zip_code)", min_value=0, value=12345, step=1)

    # --- Inputs de ubicación y estado ---
    brokered_by = st.number_input("Corredor / Agencia (brokered_by)", min_value=0.0, value=60000.0, step=100.0)
    street = st.number_input("Identificador de calle (street)", min_value=0, value=1000, step=1)

    st.markdown("</div>", unsafe_allow_html=True)

    # Botón de predicción
    predict_button = st.button("Calcular precio estimado", type="primary")


with col_result:
    st.markdown('<div class="section-title">📈 Resultado de la predicción</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if predict_button:
        # Construimos el payload tal como lo espera el modelo
        payload = {
            "bed": bed,
            "bath": bath,
            "acre_lot": acre_lot,
            "house_size": house_size,
            "zip_code": zip_code,
            "brokered_by": brokered_by,
            "street": street,
        }
        feature_df = build_feature_df(payload)

        predicted_price, raw_response = call_inference_api(payload)

        if predicted_price is not None:
            st.success("Predicción generada correctamente ✅")
            st.markdown(
                f"""
                ### Precio estimado

                <span style="font-size: 28px; font-weight: 700; color: {PRIMARY};">
                ${predicted_price:,.0f}
                </span>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                Esta estimación se basa en el modelo de regresión entrenado
                con los datos históricos de precios de vivienda del proyecto.
                """,
            )

            # Explicabilidad con SHAP
            st.markdown("---")
            st.markdown("#### 🔍 Explicabilidad (SHAP)")
            try:
                model, explainer = load_model_and_explainer()
                shap_vals, base_value = compute_shap_values(explainer, feature_df)
                shap_df = (
                    pd.DataFrame({"feature": FEATURE_COLUMNS, "contribution": shap_vals})
                    .sort_values("contribution", key=abs, ascending=False)
                )
                st.caption("Impacto de cada feature en la predicción (mayor a menor).")
                st.bar_chart(shap_df.set_index("feature"))
                st.caption(f"Valor base del modelo: ${base_value:,.0f}")
            except Exception as e:
                st.warning(f"No se pudo calcular SHAP: {e}")

            with st.expander("Ver detalles técnicos de la respuesta"):
                st.json(raw_response)

        else:
            st.error("No se pudo obtener una predicción desde la API de inferencia.")
            st.markdown("**Detalle técnico de la respuesta / error:**")
            st.json(raw_response)
    else:
        st.info("Completa los datos de la vivienda y pulsa **“Calcular precio estimado”** para ver el resultado.")

    st.markdown("</div>", unsafe_allow_html=True)

# SECCIÓN INFORMATIVA (ABAJO)
st.markdown('<div class="section-title"> Nota sobre el modelo</div>', unsafe_allow_html=True)
st.markdown(
    """
Este modelo fue entrenado como parte de un pipeline de **MLOps** que incluye:

- Ingesta diaria de datos desde la API del profesor (Airflow).
- Limpieza y transformación de los datos en MySQL (`clean_house_prices`).
- Entrenamiento y versionado de modelos en **MLflow**.
- Despliegue del mejor modelo en la **Inference API**.
- Pruebas de carga con **Locust** y monitoreo con **Prometheus + Grafana**.

La predicción mostrada es una **estimación** basada en patrones históricos;
no reemplaza una tasación profesional.
"""
)


def get_feature_names():
    """Compat helper used by tests to validate the feature contract."""
    return shared_get_feature_names()
