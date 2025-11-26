# tests/unit/contracts/test_feature_contract.py
import inspect

from dags.house_prices_train_model import train_house_price_model
from inference_api.main import HouseFeatures
from streamlit_ui.app import get_feature_names


def test_feature_columns_contract():
    # 1) desde el DAG de entrenamiento
    # Extraemos feature_cols leyendo el código (más robusto sería exportar una constante)
    from dags.house_prices_train_model import train_house_price_model

    # Para este ejemplo, vamos a reimportar feature_cols si la expones como constante.
    # Mejor: define en house_prices_train_model.py:
    # FEATURE_COLS = [...]
    from dags.house_prices_train_model import FEATURE_COLS as TRAIN_FEATURES

    # 2) desde el modelo Pydantic de la API
    api_fields = list(HouseFeatures.model_fields.keys())

    # 3) desde Streamlit
    streamlit_features = get_feature_names()

    assert TRAIN_FEATURES == api_fields == streamlit_features
