# airflow/dags/diabetes_training_pipeline.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import os, json
import pandas as pd
import numpy as np
import sqlalchemy as sa

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, accuracy_score
)

# ===== ENV =====
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
DATA_DB_URI         = os.getenv('DATA_DB_URI', 'mysql+pymysql://mlflow_user:mlflow_pass@mysql:3306/datasets_db')
EXPERIMENT_NAME     = os.getenv('DIABETES_EXPERIMENT', 'diabetes_readmission_stage2')

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'start_date': datetime(2025, 10, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

dag = DAG(
    'diabetes_training_pipeline',
    default_args=default_args,
    description='Train/Track/Register/Promote model for Diabetes readmission',
    schedule_interval='@once',   # change to cron later
    catchup=False,
    tags=['mlops','diabetes','training','stage2'],
)

def _engine():
    return sa.create_engine(DATA_DB_URI, pool_pre_ping=True, future=True)

def load_curated():
    with _engine().begin() as conn:
        df = pd.read_sql(sa.text("SELECT * FROM datasets_db.diabetes_curated"), conn)
    return df

def train_models(**kwargs):
    df = load_curated()
    if df.empty:
        raise ValueError("diabetes_curated is empty")

    # Normalize '?' to NaN just in case
    df = df.replace('?', np.nan)

    # Build target
    if 'readmitted_30' not in df.columns:
        # fallback if Stage-1 didn’t add it
        def map_readm(x): 
            if pd.isna(x): return 0
            return 1 if str(x).strip() == '<30' else 0
        df['readmitted_30'] = df['readmitted'].apply(map_readm)

    y = df['readmitted_30'].astype(int)

    # columns
    drop_cols = ['readmitted_30', 'readmitted', 'processed_at', 'ingested_at', 'batch']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # identify numeric/categorical
    # numeric candidates present in most versions:
    likely_num = [
        'time_in_hospital','num_lab_procedures','num_procedures','num_medications',
        'number_outpatient','number_emergency','number_inpatient','number_diagnoses'
    ]
    num_cols = [c for c in likely_num if c in X.columns]
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=10), cat_cols),
        ],
        remainder='drop'
    )

    models = {
        'logreg': LogisticRegression(max_iter=2000, class_weight='balanced'),
        'gb':     GradientBoostingClassifier(n_estimators=300),
        'rf':     RandomForestClassifier(n_estimators=500, class_weight='balanced_subsample', n_jobs=-1),
    }

    mlflow.set_experiment(EXPERIMENT_NAME)
    results = {}

    # split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    for name, clf in models.items():
        pipe = Pipeline(steps=[('pre', pre), ('clf', clf)])
        with mlflow.start_run(run_name=f"diabetes_{name}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"):
            pipe.fit(X_train, y_train)

            # eval on valid
            yv_pred = pipe.predict(X_valid)
            if hasattr(pipe, 'predict_proba'):
                yv_proba = pipe.predict_proba(X_valid)[:,1]
            else:
                # fallback: decision_function if available -> squish
                yv_proba = None

            metrics = {
                "acc_valid": accuracy_score(y_valid, yv_pred),
                "f1w_valid": f1_score(y_valid, yv_pred, average='weighted'),
            }
            if yv_proba is not None:
                metrics["roc_auc_valid"] = roc_auc_score(y_valid, yv_proba)
                metrics["ap_valid"]      = average_precision_score(y_valid, yv_proba)

            # test metrics (logged but selection uses valid)
            yt_pred = pipe.predict(X_test)
            mt_test = {
                "acc_test": accuracy_score(y_test, yt_pred),
                "f1w_test": f1_score(y_test, yt_pred, average='weighted'),
            }
            if yv_proba is not None:
                yt_proba = pipe.predict_proba(X_test)[:,1]
                mt_test["roc_auc_test"] = roc_auc_score(y_test, yt_proba)
                mt_test["ap_test"]      = average_precision_score(y_test, yt_proba)
            metrics.update(mt_test)

            mlflow.log_params({
                "model_type": name,
                "n_features_raw": int(X.shape[1]),
                "n_train": int(len(X_train)),
                "n_valid": int(len(X_valid)),
                "n_test": int(len(X_test)),
            })
            mlflow.log_metrics(metrics)

            # signature: raw feature-frame is fine; pipeline handles transforms
            input_example = X_valid.head(1)
            mlflow.sklearn.log_model(
                pipe, artifact_path="model",
                registered_model_name=f"diabetes_readmit_{name}",
                input_example=input_example
            )
            results[name] = metrics

    kwargs['ti'].xcom_push(key='results_json', value=json.dumps(results))
    return f"trained: {list(results.keys())}"

def select_and_register(**kwargs):
    ti = kwargs['ti']
    results = json.loads(ti.xcom_pull(key='results_json', task_ids='train_models'))
    # choose by primary metric (prefer roc_auc_valid if present; else f1w_valid)
    def score_of(m):
        r = results[m]
        return r.get('roc_auc_valid', r.get('f1w_valid', 0.0))

    best_name = max(results.keys(), key=score_of)
    best_score = score_of(best_name)
    target_registered_name = f"diabetes_readmit_{best_name}"

    client = MlflowClient()
    latest = client.get_latest_versions(target_registered_name, stages=[])
    if not latest:
        return f"no registry versions found for {target_registered_name}"

    mv_new = sorted(latest, key=lambda m: int(m.version))[-1]

    # compare vs Production
    prod = client.get_latest_versions(target_registered_name, stages=["Production"])
    if prod:
        mv_prod = prod[0]
        run = mlflow.get_run(mv_prod.run_id)
        old = run.data.metrics.get('roc_auc_valid', run.data.metrics.get('f1w_valid'))
        if old is not None and best_score <= old + 1e-6:
            client.set_model_version_tag(
                name=mv_new.name, version=mv_new.version,
                key="promotion_decision",
                value=f"NO_PROMOTE best={best_score:.4f} <= prod={old:.4f}"
            )
            return f"No promote {mv_new.name} v{mv_new.version}"
    # promote
    client.transition_model_version_stage(
        name=mv_new.name, version=mv_new.version,
        stage="Production", archive_existing_versions=True
    )
    client.set_model_version_tag(
        name=mv_new.name, version=mv_new.version,
        key="promotion_decision",
        value=f"PROMOTED best={best_score:.4f}"
    )
    return f"Promoted {mv_new.name} v{mv_new.version}"

t_train = PythonOperator(task_id='train_models', python_callable=train_models, dag=dag)
t_eval  = PythonOperator(task_id='select_and_register', python_callable=select_and_register, dag=dag)
t_train >> t_eval
 