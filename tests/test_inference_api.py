import pytest
from fastapi import HTTPException

from inference_api.main import (
    HouseFeatures,
    MODEL_NAME,
    MODEL_STAGE,
    health,
    predict,
)
from shared.feature_contract import FEATURE_COLUMNS


def test_health_ok(mocker):
    """El endpoint /health responde ok cuando el modelo carga."""
    mocker.patch(
        "inference_api.main.load_model",
        return_value={
            "model": object(),
            "model_name": MODEL_NAME,
            "model_version": "1",
            "run_id": "run123",
        },
    )
    data = health()
    assert data["status"] == "ok"
    assert data["model_name"] == MODEL_NAME
    assert data["stage"] == MODEL_STAGE


def test_health_failure(mocker):
    mocker.patch("inference_api.main.load_model", side_effect=RuntimeError("boom"))
    with pytest.raises(HTTPException) as exc:
        health()
    assert exc.value.status_code == 500
    assert "boom" in str(exc.value.detail)


def test_predict_uses_feature_contract(mocker):
    """Predicción correcta con payload válido según contract."""
    class DummyModel:
        def predict(self, df):
            assert list(df.columns) == FEATURE_COLUMNS
            return [111111.0]

    mocker.patch(
        "inference_api.main.load_model",
        return_value={
            "model": DummyModel(),
            "model_name": MODEL_NAME,
            "model_version": "1",
            "run_id": "run123",
        },
    )

    payload = HouseFeatures(**{col: 1 for col in FEATURE_COLUMNS})
    resp = predict(payload)
    assert hasattr(resp, "predicted_price")
    assert isinstance(resp.predicted_price, float)


def test_predict_inference_error(mocker):
    class DummyModel:
        def predict(self, df):
            raise ValueError("bad things")

    mocker.patch(
        "inference_api.main.load_model",
        return_value={
            "model": DummyModel(),
            "model_name": MODEL_NAME,
            "model_version": "1",
            "run_id": "run123",
        },
    )
    payload = HouseFeatures(**{col: 1 for col in FEATURE_COLUMNS})
    with pytest.raises(HTTPException) as exc:
        predict(payload)
    assert exc.value.status_code == 500
    assert "Inference error" in str(exc.value.detail)
