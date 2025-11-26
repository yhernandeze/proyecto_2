# tests/unit/api/test_main_api.py
import pytest
from fastapi.testclient import TestClient

# Importamos la app y las funciones desde inference_api/main.py
from inference_api.main import app, load_model, MODEL_NAME, MODEL_STAGE

client = TestClient(app)


def test_health_ok(mocker):
    """El endpoint /health devuelve 200 y el payload correcto cuando el modelo carga bien."""
    # Mock de load_model para que no intente hablar con MLflow real
    mock_model = object()
    mocker.patch("inference_api.main.load_model", return_value=mock_model)

    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_name"] == MODEL_NAME
    assert data["stage"] == MODEL_STAGE


def test_health_model_fails(mocker):
    """Si load_model lanza error, /health debe responder 500."""
    mocker.patch("inference_api.main.load_model", side_effect=RuntimeError("boom"))

    resp = client.get("/health")
    assert resp.status_code == 500
    assert "boom" in resp.json()["detail"]


def test_predict_ok(mocker):
    """Predicción correcta con payload válido."""
    # Mock del modelo MLflow
    class DummyModel:
        def predict(self, df):
            # df debe tener las columnas esperadas
            assert list(df.columns) == [
                "bed",
                "bath",
                "acre_lot",
                "house_size",
                "zip_code",
                "brokered_by",
                "street",
            ]
            return [123456.78]

    mocker.patch("inference_api.main.load_model", return_value=DummyModel())

    payload = {
        "bed": 3,
        "bath": 2,
        "acre_lot": 0.25,
        "house_size": 1200,
        "zip_code": 1001,
        "brokered_by": 67455,
        "street": 1698080,
    }

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "predicted_price" in data
    assert isinstance(data["predicted_price"], float)


def test_predict_inference_error(mocker):
    """Si model.predict lanza error, el endpoint debe devolver 500."""
    class DummyModel:
        def predict(self, df):
            raise ValueError("some internal error")

    mocker.patch("inference_api.main.load_model", return_value=DummyModel())

    payload = {
        "bed": 3,
        "bath": 2,
        "acre_lot": 0.25,
        "house_size": 1200,
        "zip_code": 1001,
        "brokered_by": 67455,
        "street": 1698080,
    }

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 500
    assert "Inference error" in resp.json()["detail"]
