from datetime import datetime, timedelta
import os
import json
import requests
import pymysql

from airflow import DAG
from airflow.operators.python import PythonOperator

DATA_API_URL = os.getenv("DATA_API_URL")
GROUP_NUMBER = os.getenv("GROUP_NUMBER", "6")

DATA_DB_URI = os.getenv("DATA_DB_URI")  # mysql+pymysql://user:pass@mysql:3306/datasets_db

# Simple parser for DATA_DB_URI
def _parse_mysql_uri(uri: str):
    # format: mysql+pymysql://user:pass@host:port/dbname
    prefix, rest = uri.split("://", 1)
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

def fetch_batch_from_api(**context):
    """
    Call the professor's API for the next batch of house prices.
    We assume there is some counter or the API keeps track of sequential batches by group.
    """
    if not DATA_API_URL:
        raise ValueError("DATA_API_URL is not set")

    url = f"{DATA_API_URL}/data?group={GROUP_NUMBER}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    # Store in XCom so the next task can insert into MySQL
    return data

def insert_raw_batch(**context):
    ti = context["ti"]
    batch_data = ti.xcom_pull(task_ids="fetch_batch")

    # Here we create a batch_id based on execution date or some incremental logic
    # For now, we'll just use the logical date as a hash-ish
    logical_date: datetime = context["logical_date"]
    batch_id = int(logical_date.timestamp())

    conn = _get_mysql_connection()
    try:
        with conn.cursor() as cursor:
            sql = """
                INSERT INTO raw_house_prices (batch_id, raw_payload)
                VALUES (%s, %s)
            """
            cursor.execute(sql, (batch_id, json.dumps(batch_data)))
        conn.commit()
    finally:
        conn.close()

# ----- Airflow DAG definition -----
default_args = {
    "owner": "mlops_student",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="house_prices_ingest_raw",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",  # we'll adjust later if needed
    catchup=False,
    tags=["mlops", "house_prices"],
) as dag:

    fetch_batch = PythonOperator(
        task_id="fetch_batch",
        python_callable=fetch_batch_from_api,
        provide_context=True,
    )

    insert_raw = PythonOperator(
        task_id="insert_raw",
        python_callable=insert_raw_batch,
        provide_context=True,
    )

    fetch_batch >> insert_raw
