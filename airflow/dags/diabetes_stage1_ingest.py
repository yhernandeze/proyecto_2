# airflow/dags/diabetes_stage1_ingest.py
from datetime import datetime, timedelta
import os
import pandas as pd
import numpy as np
import requests
import sqlalchemy as sa

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

DATA_API_URL = os.getenv("DATA_API_URL", "http://data_api:8080")
DATA_DB_URI = os.getenv("DATA_DB_URI", "mysql+pymysql://mlflow_user:mlflow_pass@mysql:3306/datasets_db")

default_args = {
    "owner": "mlops_team",
    "depends_on_past": False,
    "start_date": datetime(2025, 10, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

dag = DAG(
    "diabetes_stage1_ingest",
    default_args=default_args,
    description="Every 5 min: pull 15k diabetes rows from API, clean/dedupe, write to silver, trigger training",
    schedule_interval="*/5 * * * *",
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "diabetes", "stage1"],
)

def _engine():
    return sa.create_engine(DATA_DB_URI, pool_pre_ping=True, future=True)

def _table_exists(conn, schema: str, table: str) -> bool:
    return conn.execute(
        sa.text("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema=:s AND table_name=:t
        """),
        {"s": schema, "t": table}
    ).scalar_one() > 0

def fetch_batch(**context):
    resp = requests.get(f"{DATA_API_URL}/data", timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    rows = payload.get("data", [])
    if not rows:
        context["ti"].xcom_push(key="has_data", value=False)
        return "no data"
    df = pd.DataFrame(rows)
    context["ti"].xcom_push(key="has_data", value=True)
    context["ti"].xcom_push(key="batch_df_json", value=df.to_json(orient="records"))
    return f"fetched {len(df)} rows"

def clean_and_upsert(**context):
    ti = context["ti"]
    has_data = ti.xcom_pull(key="has_data", task_ids="fetch_batch")
    if not has_data:
        return "skipped (no data)"

    df_json = ti.xcom_pull(key="batch_df_json", task_ids="fetch_batch")
    df = pd.read_json(df_json, orient="records")

    # normalize '?'
    df = df.replace("?", np.nan)

    # build target
    if "readmitted" in df.columns:
        df["readmitted_30"] = df["readmitted"].apply(
            lambda x: 1 if isinstance(x, str) and x.strip() == "<30" else 0
        )
    else:
        df["readmitted_30"] = 0

    df["processed_at"] = pd.Timestamp.utcnow()

    eng = _engine()
    with eng.begin() as conn:
        exists = _table_exists(conn, "datasets_db", "diabetes_curated")

    # if table does NOT exist, create it with this df schema
    if not exists:
        # create empty table with correct columns
        with eng.begin() as conn:
            df.head(0).to_sql(
                "diabetes_curated",
                con=conn,
                schema="datasets_db",
                if_exists="replace",
                index=False,
            )

    # (optional) lightweight dedupe in pandas if encounter_id present
    if "encounter_id" in df.columns:
        df = df.drop_duplicates(subset=["encounter_id"], keep="last")

    # append all rows
    with eng.begin() as conn:
        df.to_sql(
            "diabetes_curated",
            con=conn,
            schema="datasets_db",
            if_exists="append",
            index=False,
        )

    return f"curated {len(df)} rows"

fetch_task = PythonOperator(
    task_id="fetch_batch",
    python_callable=fetch_batch,
    dag=dag,
)

clean_task = PythonOperator(
    task_id="clean_and_upsert",
    python_callable=clean_and_upsert,
    dag=dag,
)

trigger_train = TriggerDagRunOperator(
    task_id="trigger_training",
    trigger_dag_id="diabetes_training_pipeline",
    dag=dag,
)

fetch_task >> clean_task >> trigger_train
 