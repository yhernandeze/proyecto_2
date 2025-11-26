from datetime import datetime, timedelta
import os
import json
import pymysql

from airflow import DAG
from airflow.operators.python import PythonOperator

# Reutilizamos las mismas variables de entorno que el DAG de ingest
DATA_DB_URI = os.getenv("DATA_DB_URI")  # mysql+pymysql://user:pass@mysql:3306/datasets_db


def _parse_mysql_uri(uri: str):
    """
    Parse a MySQL URI of the form:
      mysql+pymysql://user:pass@host:port/dbname
    into a dict with connection params.
    """
    if not uri:
        raise ValueError("DATA_DB_URI is not set")

    prefix, rest = uri.split("://", 1)  # ignore 'mysql+pymysql'
    creds, hostpart = rest.split("@", 1)
    user, password = creds.split(":", 1)
    hostport, dbname = hostpart.split("/", 1)

    if ":" in hostport:
        host, port = hostport.split(":", 1)
        port = int(port)
    else:
        host, port = hostport, 3306

    return {
        "user": user,
        "password": password,
        "host": host,
        "port": port,
        "database": dbname,
    }


def _get_mysql_connection():
    cfg = _parse_mysql_uri(DATA_DB_URI)
    return pymysql.connect(
        host=cfg["host"],
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
        port=cfg["port"],
        cursorclass=pymysql.cursors.DictCursor,
    )


def build_clean_from_latest_batch(**context):
    """
    Lee el último registro de raw_house_prices,
    extrae raw_payload.data[*] y lo inserta en clean_house_prices.
    """
    conn = _get_mysql_connection()
    try:
        with conn.cursor() as cursor:
            # 1) Obtener el último batch almacenado en raw
            cursor.execute(
                """
                SELECT id, batch_id, raw_payload, received_at
                FROM raw_house_prices
                ORDER BY received_at DESC, id DESC
                LIMIT 1
                """
            )
            row = cursor.fetchone()
            if not row:
                print("No hay filas en raw_house_prices. Nada que limpiar.")
                return

            batch_id = row["batch_id"]
            received_at = row["received_at"]
            print(f"Usando batch_id={batch_id}, received_at={received_at}")

            # 2) Revisar si ya existe ese batch en clean (idempotencia básica)
            cursor.execute(
                """
                SELECT COUNT(*) AS n
                FROM clean_house_prices
                WHERE batch_id = %s
                """,
                (batch_id,),
            )
            already = cursor.fetchone()["n"]
            if already > 0:
                print(f"Ya existen {already} filas en clean_house_prices para batch_id={batch_id}. No se inserta nada.")
                return

            # 3) Parsear el JSON
            payload = row["raw_payload"]
            if isinstance(payload, str):
                payload = json.loads(payload)

            group_number = payload.get("group_number")
            api_day = payload.get("day")
            api_batch_number = payload.get("batch_number")
            records = payload.get("data", [])

            print(
                f"Payload: group_number={group_number}, day={api_day}, "
                f"api_batch_number={api_batch_number}, registros={len(records)}"
            )

            if not records:
                print("No hay registros en payload['data']. Nada que insertar.")
                return

            # 4) Preparar los datos "limpios"
            insert_sql = """
                INSERT INTO clean_house_prices (
                    batch_id,
                    group_number,
                    api_day,
                    api_batch_number,
                    brokered_by,
                    status,
                    price,
                    bed,
                    bath,
                    acre_lot,
                    street,
                    city,
                    state,
                    zip_code,
                    house_size,
                    prev_sold_date
                )
                VALUES (
                    %(batch_id)s,
                    %(group_number)s,
                    %(api_day)s,
                    %(api_batch_number)s,
                    %(brokered_by)s,
                    %(status)s,
                    %(price)s,
                    %(bed)s,
                    %(bath)s,
                    %(acre_lot)s,
                    %(street)s,
                    %(city)s,
                    %(state)s,
                    %(zip_code)s,
                    %(house_size)s,
                    %(prev_sold_date)s
                )
            """

            clean_rows = []
            for rec in records:
                clean_rows.append(
                    {
                        "batch_id": batch_id,
                        "group_number": group_number,
                        "api_day": api_day,
                        "api_batch_number": api_batch_number,
                        "brokered_by": rec.get("brokered_by"),
                        "status": rec.get("status"),
                        "price": rec.get("price"),
                        "bed": rec.get("bed"),
                        "bath": rec.get("bath"),
                        "acre_lot": rec.get("acre_lot"),
                        "street": rec.get("street"),
                        "city": rec.get("city"),
                        "state": rec.get("state"),
                        "zip_code": rec.get("zip_code"),
                        "house_size": rec.get("house_size"),
                        "prev_sold_date": rec.get("prev_sold_date"),
                    }
                )

            print(f"Insertando {len(clean_rows)} filas en clean_house_prices...")
            cursor.executemany(insert_sql, clean_rows)
        conn.commit()
        print("Inserción completada correctamente.")
    finally:
        conn.close()


# ----- Definición del DAG -----
default_args = {
    "owner": "mlops_student",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="house_prices_build_clean",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,  # lo lanzamos manual al principio
    catchup=False,
    tags=["mlops", "house_prices", "clean"],
) as dag:

    build_clean = PythonOperator(
        task_id="build_clean_from_latest_batch",
        python_callable=build_clean_from_latest_batch,
        provide_context=True,
    )
