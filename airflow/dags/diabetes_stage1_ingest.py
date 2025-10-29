# airflow/dags/diabetes_stage1_ingest.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.models import Variable
import os, time
import pandas as pd
import numpy as np
import sqlalchemy as sa

# optional: download via ucimlrepo
from ucimlrepo import fetch_ucirepo

# ===== ENV =====
DATA_DB_URI = os.getenv('DATA_DB_URI', 'mysql+pymysql://mlflow_user:mlflow_pass@mysql:3306/datasets_db')
BATCH_SIZE = int(os.getenv('DIABETES_BATCH_SIZE', '15000'))

default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'start_date': datetime(2025, 10, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

dag = DAG(
    'diabetes_stage1_ingest',
    default_args=default_args,
    description='Download Diabetes CSV -> RAW (batches) -> CURATED (by batch)',
    schedule_interval='*/6 * * * *',
    catchup=False,
    tags=['mlops','diabetes','stage1'],
)

def _engine():
    return sa.create_engine(DATA_DB_URI, pool_pre_ping=True, future=True)

def _table_exists(conn, schema, table):
    return conn.execute(
        sa.text("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema=:s AND table_name=:t
        """),
        {"s": "datasets_db", "t": table}
    ).scalar_one() > 0

def _mark_done():
    Variable.set("diabetes_harvest_done", "true")

def stop_if_done():
    done = Variable.get("diabetes_harvest_done", default_var="false").lower() == "true"
    return not done

task_gate = ShortCircuitOperator(
    task_id='stop_if_done',
    python_callable=stop_if_done,
    dag=dag
)

def download_full_csv(**kwargs):
    """
    Download the Diabetes dataset once and cache in MySQL (staging table),
    then we will slice into batches in the next task.
    """
    # 1) try to detect if staging already exists
    with _engine().begin() as conn:
        if _table_exists(conn, "datasets_db", "diabetes_staging_full"):
            return "staging exists"

    # 2) download via ucimlrepo (id=296)
    ds = fetch_ucirepo(id=296)
    X = ds.data.features.copy()  # DataFrame
    y = ds.data.targets.copy() if hasattr(ds.data, 'targets') else None

    # Combine to one DataFrame; the UCI repo usually exposes single table in X
    df = X.copy()
    if y is not None and isinstance(y, pd.DataFrame):
        # join if targets provided
        for c in y.columns:
            if c not in df.columns:
                df[c] = y[c]

    # Standardize column names (lower, underscores)
    df.columns = [c.strip().replace(' ', '_') for c in df.columns]

    # Persist whole CSV once (no batch here; batch in next task)
    with _engine().begin() as conn:
        df.to_sql('diabetes_staging_full', con=conn, schema='datasets_db', if_exists='replace', index=False)

    return f"downloaded {len(df)} rows to staging"

def ingest_next_batch(**kwargs):
    """
    Move next chunk of up to BATCH_SIZE rows from staging -> diabetes_raw,
    tagging with batch number.
    When no more rows, mark done & short-circuit next runs.
    """
    ti = kwargs['ti']
    with _engine().begin() as conn:
        # compute next batch number
        if _table_exists(conn, "datasets_db", "diabetes_raw"):
            max_batch = conn.execute(sa.text("SELECT COALESCE(MAX(batch),0) FROM datasets_db.diabetes_raw")).scalar_one()
        else:
            max_batch = 0
        next_batch = int(max_batch) + 1

        # compute offset
        already = conn.execute(sa.text("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='datasets_db' AND table_name='diabetes_raw'")).scalar_one()
        # rows already moved:
        moved = 0
        if _table_exists(conn, "datasets_db", "diabetes_raw"):
            moved = conn.execute(sa.text("SELECT COUNT(*) FROM datasets_db.diabetes_raw")).scalar_one()

        total = conn.execute(sa.text("SELECT COUNT(*) FROM datasets_db.diabetes_staging_full")).scalar_one()
        if int(moved) >= int(total):
            _mark_done()
            raise Exception("No more rows to ingest")

        # select window [moved, moved+BATCH_SIZE)
        df = pd.read_sql(
            sa.text("SELECT * FROM datasets_db.diabetes_staging_full LIMIT :lim OFFSET :off"),
            conn, params={"lim": int(BATCH_SIZE), "off": int(moved)}
        )
        if df.empty:
            _mark_done()
            raise Exception("No more rows to ingest (empty window)")

        df['batch'] = next_batch
        df['ingested_at'] = pd.Timestamp.utcnow()
        df.to_sql('diabetes_raw', con=conn, schema='datasets_db', if_exists='append', index=False)

    ti.xcom_push(key='last_batch', value=next_batch)
    return f"ingested raw batch={next_batch} rows={len(df)}"

def curate_last_batch(**kwargs):
    """
    Minimal clean/encode for Stage-2 training:
    - build readmitted_30 from readmitted
    - replace '?' with NaN in known cols
    - drop ids
    - keep core numeric; keep cats for OHE later
    """
    from airflow.exceptions import AirflowSkipException
    ti = kwargs['ti']
    last_batch = ti.xcom_pull(key='last_batch', task_ids='ingest_next_batch')
    if last_batch is None:
        raise AirflowSkipException("no new batch")

    with _engine().begin() as conn:
        raw = pd.read_sql(
            sa.text("SELECT * FROM datasets_db.diabetes_raw WHERE batch=:b"),
            conn, params={"b": int(last_batch)}
        )

    if raw.empty:
        raise AirflowSkipException(f"batch={last_batch} empty")

    # normalize '?' to NaN
    raw = raw.replace('?', np.nan)

    # target
    def map_readmitted(x):
        if pd.isna(x): return 0
        return 1 if str(x).strip() == '<30' else 0
    raw['readmitted_30'] = raw['readmitted'].apply(map_readmitted)

    # drop ids
    drop_cols = [c for c in ['encounter_id','patient_nbr'] if c in raw.columns]
    curated = raw.drop(columns=drop_cols, errors='ignore').copy()

    # stamp
    curated['processed_at'] = pd.Timestamp.utcnow()

    with _engine().begin() as conn:
        # upsert-by-batch (delete then append)
        if _table_exists(conn, "datasets_db", "diabetes_curated"):
            conn.execute(sa.text("DELETE FROM datasets_db.diabetes_curated WHERE batch=:b"), {"b": int(last_batch)})
        curated.to_sql('diabetes_curated', con=conn, schema='datasets_db', if_exists='append', index=False)

    return f"curated batch={last_batch} rows={len(curated)}"

t_download = PythonOperator(task_id='download_full_csv', python_callable=download_full_csv, dag=dag)
t_ingest   = PythonOperator(task_id='ingest_next_batch', python_callable=ingest_next_batch, dag=dag)
t_curate   = PythonOperator(task_id='curate_last_batch', python_callable=curate_last_batch, dag=dag)

task_gate >> t_download >> t_ingest >> t_curate