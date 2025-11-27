import os
import requests
import streamlit as st
from datetime import datetime

# -----------------------------
# Configuración de la página
# -----------------------------
st.set_page_config(
    page_title="MLOps Dashboard - House Prices",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Estilos (colores institucionales)
# Azul / teal / gris claro
# -----------------------------
PRIMARY = "#004A9F"   # azul institucional
ACCENT = "#00A19B"    # teal
BG_LIGHT = "#F5F7FB"  # gris muy claro
CARD_BG = "#FFFFFF"   # blanco

st.markdown(
    f"""
    <style>
    .main {{
        background-color: {BG_LIGHT};
    }}
    .big-title {{
        font-size: 32px;
        font-weight: 700;
        color: {PRIMARY};
        margin-bottom: 0.2rem;
    }}
    .subtitle {{
        font-size: 14px;
        color: #4F4F4F;
        margin-bottom: 1.5rem;
    }}
    .metric-card {{
        background-color: {CARD_BG};
        padding: 1.2rem 1.0rem;
        border-radius: 10px;
        box-shadow: 0 0 8px rgba(15, 23, 42, 0.05);
        border-left: 4px solid {ACCENT};
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

# Sidebar: info de proyecto
st.sidebar.title("Proyecto MLOps")
st.sidebar.markdown(
    """
**Pipeline general**

1. Ingesta desde API (Airflow)  
2. Almacenamiento en MySQL (`raw_house_prices`)  
3. Limpieza y feature engineering  
4. Entrenamiento & registro en **MLflow**  
5. Despliegue del modelo en **Inference API**  
6. Pruebas de carga con **Locust**  
7. Monitoreo con **Prometheus + Grafana**  
8. Visualización en **Streamlit**
"""
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Variables de entorno detectadas:**")

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "no configurado")
inference_url = os.getenv("INFERENCE_API_URL", "http://localhost:8989")
group_number = os.getenv("GROUP_NUMBER", "–")

st.sidebar.markdown(f"- `MLFLOW_TRACKING_URI`: `{mlflow_uri}`")
st.sidebar.markdown(f"- `INFERENCE_API_URL`: `{inference_url}`")
st.sidebar.markdown(f"- `GROUP_NUMBER`: `{group_number}`")

# Encabezado principal

st.markdown('<div class="big-title">MLOps Dashboard – House Prices</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Vista general del flujo de ingesta, entrenamiento y despliegue del modelo de precios de vivienda.</div>',
    unsafe_allow_html=True,
)

# Métricas principales (placeholder ligeros)

col1, col2, col3, col4 = st.columns(4)

with col1:
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.caption("Lotes ingeridos (Airflow → API profesor)")
        st.metric("Batches totales", value="12")
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.caption("Modelos entrenados (MLflow)")
        st.metric("Runs registrados", value="5")
        st.markdown("</div>", unsafe_allow_html=True)

with col3:
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.caption("Modelo en producción")
        st.metric("Versión", value="v2.1", delta="RandomForestRegressor")
        st.markdown("</div>", unsafe_allow_html=True)

with col4:
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.caption("Último entrenamiento")
        st.metric("Fecha", value=datetime.now().strftime("%Y-%m-%d"))
        st.markdown("</div>", unsafe_allow_html=True)


# Estado de servicios (health checks muy ligeros)

st.markdown('<div class="section-title">🔌 Estado de servicios</div>', unsafe_allow_html=True)
st.markdown('<p class="small-label">Se realizan verificaciones muy ligeras contra los endpoints de salud.</p>', unsafe_allow_html=True)

svc_col1, svc_col2, svc_col3 = st.columns(3)

def check_health(url: str) -> str:
    try:
        resp = requests.get(url, timeout=2)
        if resp.status_code == 200:
            return " OK"
        return f" {resp.status_code}"
    except Exception:
        return "No disponible"

with svc_col1:
    st.markdown("**Inference API**")
    inf_health = check_health(f"{inference_url.rstrip('/')}/health")
    st.write(f"Estado: {inf_health}")
    st.caption(f"URL: {inference_url}/health")

with svc_col2:
    st.markdown("**MLflow Tracking**")
    if mlflow_uri != "no configurado":
        mlflow_health = check_health(f"{mlflow_uri.rstrip('/')}/health")
        st.write(f"Estado: {mlflow_health}")
        st.caption(f"URL: {mlflow_uri}/health")
    else:
        st.write("No configurado")
        st.caption("Define `MLFLOW_TRACKING_URI` para verificar salud.")

with svc_col3:
    st.markdown("**Locust (carga)**")
    locust_url = os.getenv("LOCUST_URL", "http://localhost:8089")
    locust_health = check_health(locust_url)
    st.write(f"Estado: {locust_health}")
    st.caption(f"UI: {locust_url}")


# Resumen textual del flujo

st.markdown('<div class="section-title"> Resumen del flujo MLOps</div>', unsafe_allow_html=True)

st.markdown(
    """
Este proyecto orquesta un flujo de **MLOps completo**:

- **Ingesta**: Airflow consume la API y guarda los lotes de datos en MySQL (`raw_house_prices`).
- **Preparación**: Se limpian y transforman los datos para entrenamiento (features numéricas para regresión).
- **Entrenamiento**: Se entrena un modelo de regresión ('RandomForestRegressor`) y se registra en **MLflow**.
- **Despliegue**: El mejor modelo se carga en la **Inference API** (FastAPI) para servir predicciones.
- **Pruebas de carga**: Con **Locust** se generan peticiones concurrentes para medir latencia y throughput.
- **Monitoreo**: **Prometheus + Grafana** recolectan métricas técnicas del sistema.
- **Visualización**: Esta interfaz en **Streamlit** resume el estado general del pipeline.
"""
)

