# tests/unit/airflow/test_house_prices_ingest_raw.py
import json
from datetime import datetime

import pytest

from dags.house_prices_ingest_raw import (
    _parse_mysql_uri,
    fetch_batch_from_api,
    insert_raw_batch,
)


def test_parse_mysql_uri_basic():
    uri = "mysql+pymysql://user123:pass456@mysql:3306/datasets_db"
    cfg = _parse_mysql_uri(uri)
    assert cfg == {
        "user": "user123",
        "password": "pass456",
        "host": "mysql",
        "port": 3306,
        "database": "datasets_db",
    }


def test_fetch_batch_from_api_tuesday(mocker):
    """Verifica que se llama al endpoint correcto con params correctos."""
    mock_get = mocker.patch("dags.house_prices_ingest_raw.requests.get")

    # Respuesta fake
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.text = '{"ok": true}'
    mock_response.json.return_value = {"group_number": 6, "day": "Tuesday", "data": []}
    mock_get.return_value = mock_response

    logical_date = datetime(2025, 1, 7)  # es martes
    context = {"logical_date": logical_date}

    data = fetch_batch_from_api(**context)
    assert data["day"] == "Tuesday"
    mock_get.assert_called_once()
    called_url = mock_get.call_args.args[0]
    called_params = mock_get.call_args.kwargs["params"]
    assert "/data" in called_url
    assert called_params["group_number"] == 6
    assert called_params["day"] == "Tuesday"


def test_insert_raw_batch_inserts_json(mocker):
    """insert_raw_batch debe insertar batch_id y raw_payload como JSON en la tabla."""
    # XCom data simulado
    fake_batch = {"group_number": 6, "day": "Tuesday", "data": [{"bed": 3}]}
    ti = mocker.Mock()
    ti.xcom_pull.return_value = fake_batch

    logical_date = datetime(2025, 1, 7)
    context = {"ti": ti, "logical_date": logical_date}

    # Mock de conexión MySQL
    fake_cursor = mocker.Mock()
    fake_conn = mocker.Mock()
    fake_conn.cursor.return_value.__enter__.return_value = fake_cursor

    mocker.patch(
        "dags.house_prices_ingest_raw._get_mysql_connection",
        return_value=fake_conn,
    )

    insert_raw_batch(**context)

    fake_cursor.execute.assert_called_once()
    sql, params = fake_cursor.execute.call_args.args
    assert "INSERT INTO raw_house_prices" in sql
    # params[1] debe ser un JSON string
    payload_str = params[1]
    parsed = json.loads(payload_str)
    assert parsed["group_number"] == 6
    assert parsed["day"] == "Tuesday"
