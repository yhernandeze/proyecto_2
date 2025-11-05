# airflow/dags/diabetes_training_pipeline.py
from datetime import datetime, timedelta
import os
import json
import pandas as pd
import numpy as np
import sqlalchemy as sa
import requests

from airflow import DAG
from airflow.operators.python import PythonOperator

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, accuracy_score
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
DATA_DB_URI = os.getenv("DATA_DB_URI", "mysql+pymysql://mlflow_user:mlflow_pass@mysql:3306/datasets_db")
EXPERIMENT_NAME = os.getenv("DIABETES_EXPERIMENT", "diabetes_readmission_stage2")
INFERENCE_API_URL = os.getenv("INFERENCE_API_URL", "http://inference_api:8989")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

default_args = {
    "owner": "mlops_team",
    "depends_on_past": False,
    "start_date": datetime(2025, 10, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "diabetes_training_pipeline",
    default_args=default_args,
    description="Train logistic regression on silver; compare; promote; refresh inference",
    schedule_interval=None,   # triggered by ingest DAG
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "diabetes", "training"],
)

def _engine():
    return sa.create_engine(DATA_DB_URI, pool_pre_ping=True, future=True)

def load_curated():
    with _engine().begin() as conn:
        df = pd.read_sql(sa.text("SELECT * FROM datasets_db.diabetes_curated"), conn)
    return df

def train_and_register(**context):
    df = load_curated()
    if df.empty:
        raise ValueError("diabetes_curated is empty")

    df = df.replace("?", np.nan)

    # target
    if "readmitted_30" not in df.columns:
        df["readmitted_30"] = df["readmitted"].apply(
            lambda x: 1 if isinstance(x, str) and x.strip() == "<30" else 0
        )
    y = df["readmitted_30"].astype(int)

    drop_cols = ["readmitted_30", "readmitted", "processed_at", "ingested_at", "batch"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # numeric / categorical
    likely_num = [
        "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
        "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
        "admission_type_id", "discharge_disposition_id", "admission_source_id"
    ]
    num_cols = [c for c in likely_num if c in X.columns]
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), cat_cols),
        ],
        remainder="drop"
    )

    # split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    pipe = Pipeline(steps=[
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"diabetes_logreg_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}") as run:
        pipe.fit(X_train, y_train)

        yv_pred = pipe.predict(X_valid)
        yv_proba = pipe.predict_proba(X_valid)[:, 1]

        metrics = {
            "acc_valid": accuracy_score(y_valid, yv_pred),
            "f1w_valid": f1_score(y_valid, yv_pred, average="weighted"),
            "roc_auc_valid": roc_auc_score(y_valid, yv_proba),
            "ap_valid": average_precision_score(y_valid, yv_proba),
        }

        yt_pred = pipe.predict(X_test)
        yt_proba = pipe.predict_proba(X_test)[:, 1]
        metrics.update({
            "acc_test": accuracy_score(y_test, yt_pred),
            "f1w_test": f1_score(y_test, yt_pred, average="weighted"),
            "roc_auc_test": roc_auc_score(y_test, yt_proba),
            "ap_test": average_precision_score(y_test, yt_proba),
        })

        mlflow.log_metrics(metrics)
        mlflow.log_params({
            "model_type": "logreg",
            "n_features_raw": int(X.shape[1]),
            "n_train": int(len(X_train)),
            "n_valid": int(len(X_valid)),
            "n_test": int(len(X_test)),
        })

        model_name = "diabetes_readmit_logreg"

        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            registered_model_name=model_name,
            input_example=X_valid.head(1)
        )

        # store metrics json so next task can read
        context["ti"].xcom_push(key="new_metrics", value=json.dumps(metrics))
        context["ti"].xcom_push(key="model_name", value=model_name)
        context["ti"].xcom_push(key="run_id", value=run.info.run_id)

def promote_and_refresh(**context):
    ti = context["ti"]
    new_metrics = json.loads(ti.xcom_pull(key="new_metrics", task_ids="train_and_register"))
    model_name = ti.xcom_pull(key="model_name", task_ids="train_and_register")

    client = MlflowClient()
    # latest version we just logged:
    latest = client.get_latest_versions(model_name, stages=[])
    if not latest:
        return "no model versions found"

    mv_new = sorted(latest, key=lambda m: int(m.version))[-1]
    new_score = new_metrics.get("roc_auc_valid", new_metrics.get("f1w_valid", 0.0))

    # check current production
    prod = client.get_latest_versions(model_name, stages=["Production"])
    should_promote = True
    if prod:
        mv_prod = prod[0]
        run_prod = mlflow.get_run(mv_prod.run_id)
        old_score = run_prod.data.metrics.get("roc_auc_valid", run_prod.data.metrics.get("f1w_valid", 0.0))
        if old_score is not None and new_score <= old_score + 1e-6:
            should_promote = False

    if should_promote:
        client.transition_model_version_stage(
            name=mv_new.name,
            version=mv_new.version,
            stage="Production",
            archive_existing_versions=True
        )
        promoted = True
    else:
        promoted = False

    # tell inference_api to reload
    try:
        body = {
            "model_name": model_name,
            "stage": "Production"
        }
        r = requests.post(f"{INFERENCE_API_URL}/models/load", json=body, timeout=10)
        r.raise_for_status()
    except Exception as e:
        # log but don't fail
        print(f"Could not refresh inference API: {e}")

    return f"done; promoted={promoted}"

t_train = PythonOperator(
    task_id="train_and_register",
    python_callable=train_and_register,
    dag=dag,
)

t_promote = PythonOperator(
    task_id="promote_and_refresh",
    python_callable=promote_and_refresh,
    dag=dag,
)

t_train >> t_promote