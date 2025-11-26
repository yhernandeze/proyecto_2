# tests/unit/airflow/test_house_prices_build_clean.py
import json
from datetime import datetime

import pytest

from dags.house_prices_build_clean import build_clean_from_latest_batch


def test_build_clean_no_rows_in_raw(mocker, capsys):
    """Si no hay filas en raw_house_prices, no se inserta nada en clean."""
    fake_cursor = mocker.Mock()
    # SELECT ... FROM raw_house_prices LIMIT 1 -> None
    fake_cursor.fetchone.side_effect = [
        None,  # primera llamada: no hay filas
    ]
    fake_conn = mocker.Mock()
    fake_conn.cursor.return_value.__enter__.return_value = fake_cursor

    mocker.patch(
        "dags.house_prices_build_clean._get_mysql_connection",
        return_value=fake_conn,
    )

    build_clean_from_latest_batch()
    captured = capsys.readouterr()
    assert "No hay filas en raw_house_prices" in captured.out
    # No debe llamarse a executemany
    assert not fake_cursor.executemany.called


def test_build_clean_inserts_rows(mocker):
    """Caso feliz: hay un batch nuevo y se insertan filas limpias en clean_house_prices."""
    # 1) Row de raw_house_prices
    payload = {
        "group_number": 6,
        "day": "Tuesday",
        "batch_number": 1,
        "data": [
            {
                "brokered_by": 67455,
                "status": "for_sale",
                "price": 289900,
                "bed": 3,
                "bath": 2,
                "acre_lot": 0.36,
                "street": 1698080,
                "city": "X",
                "state": "Y",
                "zip_code": 1001,
                "house_size": 1276,
                "prev_sold_date": None,
            }
        ],
    }

    raw_row = {
        "id": 1,
        "batch_id": 12345,
        "raw_payload": json.dumps(payload),
        "received_at": datetime(2025, 1, 7, 10, 0, 0),
    }

    fake_cursor = mocker.Mock()
    # Orden de llamadas a fetchone:
    # 1) SELECT ... FROM raw_house_prices ... LIMIT 1
    # 2) SELECT COUNT(*) AS n FROM clean_house_prices WHERE batch_id = %s
    fake_cursor.fetchone.side_effect = [
        raw_row,
        {"n": 0},  # no existe aún este batch en clean
    ]

    fake_conn = mocker.Mock()
    fake_conn.cursor.return_value.__enter__.return_value = fake_cursor

    mocker.patch(
        "dags.house_prices_build_clean._get_mysql_connection",
        return_value=fake_conn,
    )

    build_clean_from_latest_batch()

    # Verificamos que se llamó a executemany con una fila
    fake_cursor.executemany.assert_called_once()
    sql, rows = fake_cursor.executemany.call_args.args
    assert "INSERT INTO clean_house_prices" in sql
    assert len(rows) == 1
    r0 = rows[0]
    assert r0["batch_id"] == 12345
    assert r0["group_number"] == 6
    assert r0["price"] == 289900
    assert r0["bed"] == 3
