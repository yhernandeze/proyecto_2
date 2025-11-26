# tests/unit/airflow/test_house_prices_train_model.py
import numpy as np
import pytest

from dags.house_prices_train_model import train_house_price_model


def test_train_house_price_model_happy_path(mocker):
    """Entrenamiento 'happy path' con datos sintéticos y mlflow mockeado."""
    # Datos sintéticos
    rows = []
    for i in range(50):
        rows.append(
            {
                "bed": 3,
                "bath": 2,
                "acre_lot": 0.2 + 0.01 * i,
                "house_size": 1000 + 10 * i,
                "zip_code": 1000 + (i % 5),
                "brokered_by": 60000 + i,
                "street": 1700000 + i,
                "price": 200000 + 1000 * i,
            }
        )

    fake_cursor = mocker.Mock()
    fake_cursor.fetchall.return_value = rows

    fake_conn = mocker.Mock()
    fake_conn.cursor.return_value.__enter__.return_value = fake_cursor

    mocker.patch(
        "dags.house_prices_train_model._get_mysql_connection",
        return_value=fake_conn,
    )

    # Mock de mlflow
    fake_run = mocker.Mock()
    fake_run.__enter__.return_value = fake_run
    fake_run.info.run_id = "test_run_id"

    mock_mlflow = mocker.patch("dags.house_prices_train_model.mlflow")
    mock_mlflow.start_run.return_value = fake_run

    # Ejecutar
    train_house_price_model()

    # Aserciones
    mock_mlflow.set_experiment.assert_called_once()
    mock_mlflow.start_run.assert_called_once()
    # Debe haber registrado algún métrico
    assert mock_mlflow.log_metric.call_count >= 2  # rmse y r2
