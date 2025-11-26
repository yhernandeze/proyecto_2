from datetime import datetime, timedelta
import os

import pandas as pd
import pymysql

from airflow import DAG
from airflow.operators.python import PythonOperator

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

import mlflow
import mlflow.sklearn

# --------- ENV VARS ----------
DATA_DB_URI = os.getenv("DATA_DB_URI")  # mysql+pymysql://user:pass@mysql:3306/datasets_db
GROUP_NUMBER = os.getenv("GROUP_NUMBER", "6")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# --------- HELPERS TO CONNECT TO MYSQL ----------
def _parse_mysql_uri(uri: str):
    """
    Very simple URI parser for mysql+pymysql://user:pass@host:port/dbname
    """
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

# --------- MAIN TRAINING FUNCTION ----------
def train_house_price_model(**context):
    """
    Load clean_house_prices, train a regression model, log to MLflow and
    register it as 'house_prices_regression'.
    """
    print("[train_house_price_model] DATA_DB_URI in task:", DATA_DB_URI)

    feature_cols = [
        "bed",
        "bath",
        "acre_lot",
        "house_size",
        "zip_code",
        "brokered_by",
        "street",
    ]
    target_col = "price"

    # 1) Load data from MySQL using cursor.fetchall() + DataFrame
    conn = _get_mysql_connection()
    try:
        query = """
            SELECT
                bed,
                bath,
                acre_lot,
                house_size,
                zip_code,
                brokered_by,
                street,
                price
            FROM clean_house_prices
            WHERE price IS NOT NULL
        """
        print("[train_house_price_model] SQL being executed:")
        print(query)

        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            print(f"[train_house_price_model] Rows fetched from DB: {len(rows)}")

        if not rows:
            raise ValueError("clean_house_prices returned no rows with non-null price.")

        df = pd.DataFrame(rows)
    finally:
        conn.close()

    print("[train_house_price_model] Raw dtypes from DB (DataFrame):")
    print(df.dtypes)
    print("[train_house_price_model] Head:")
    print(df.head().to_string())

    # 2) Sanity check: asegurarnos de que no son strings tipo 'bed', 'bath', etc.
    # (Si esto vuelve a pasar, algo upstream está mal).
    sample_values = df.head(5)[feature_cols + [target_col]]
    all_values = sample_values.values.flatten().tolist()
    suspicious = all(v in ["bed", "bath", "acre_lot", "house_size", "zip_code",
                           "brokered_by", "street", "price"] for v in set(all_values))

    if suspicious:
        raise ValueError(
            "Data in clean_house_prices looks like header strings, "
            "not real numeric values. Fix upstream ETL."
        )

    # 3) Convertir a float de forma robusta
    for col in feature_cols + [target_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print("[train_house_price_model] dtypes after to_numeric:")
    print(df.dtypes)

    before_drop = len(df)
    df = df.dropna(subset=feature_cols + [target_col])
    after_drop = len(df)
    print(
        f"[train_house_price_model] Dropped {before_drop - after_drop} rows with NaNs; "
        f"remaining {after_drop}"
    )

    if after_drop < 10:
        raise ValueError(
            f"Too few valid rows after cleaning ({after_drop}). "
            "Check clean_house_prices contents."
        )

    # 4) Build X, y
    X = df[feature_cols].to_numpy(dtype="float64")
    y = df[target_col].to_numpy(dtype="float64")

    print(f"[train_house_price_model] Shape of X: {X.shape}, y: {y.shape}")

    # 5) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 6) Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment_name = f"group_{GROUP_NUMBER}_house_prices"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="house_price_random_forest") as run:
        params = {
            "n_estimators": 200,
            "max_depth": 10,
            "random_state": 42,
        }
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        # 7) Evaluate
        y_pred = model.predict(X_test)
        rmse = sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # 8) Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_param("feature_cols", ",".join(feature_cols))

        registered_model_name = "house_prices_regression"
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=registered_model_name,
        )

        mlflow.set_tag("source", "airflow_dag_house_prices_train_model")

        print(f"Run ID: {run.info.run_id}")
        print(f"RMSE: {rmse:.2f}, R2: {r2:.4f}")

# --------- DAG DEFINITION ----------
default_args = {
    "owner": "mlops_student",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="house_prices_train_model",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,  # manual trigger
    catchup=False,
    tags=["mlops", "house_prices", "training"],
) as dag:

    train_task = PythonOperator(
        task_id="train_house_model",
        python_callable=train_house_price_model,
        provide_context=True,
    )
