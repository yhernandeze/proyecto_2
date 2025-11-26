from datetime import datetime, timedelta
import os
from math import sqrt

import pandas as pd
import pymysql

from airflow import DAG
from airflow.operators.python import PythonOperator

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# --------- ENV VARS ----------
DATA_DB_URI = os.getenv("DATA_DB_URI")  # mysql+pymysql://user:pass@mysql:3306/datasets_db
GROUP_NUMBER = os.getenv("GROUP_NUMBER", "6")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

REGISTERED_MODEL_NAME = "house_prices_regression"

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
    Load clean_house_prices, decide whether to retrain based on
    new data / drift, train if needed, and (optionally) promote in MLflow.
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

    # -------------------- 0) GLOBAL STATS + DECISION --------------------
    conn = _get_mysql_connection()
    last_meta = None
    with conn.cursor() as cur:
        # Stats globales de clean_house_prices
        cur.execute(
            """
            SELECT
              COUNT(*) AS total_rows,
              AVG(price) AS mean_price,
              VAR_POP(price) AS var_price,
              MAX(batch_id) AS max_batch_id
            FROM clean_house_prices
            WHERE price IS NOT NULL
            """
        )
        stats = cur.fetchone()
        total_rows = stats["total_rows"] or 0
        mean_price = stats["mean_price"] or 0.0
        var_price = stats["var_price"] or 0.0
        max_batch_id = stats["max_batch_id"]

        print("[train_house_price_model] Global stats:",
              f"total_rows={total_rows}, mean_price={mean_price}, var_price={var_price}, max_batch_id={max_batch_id}")

        # Último registro de training_runs, si existe
        cur.execute(
            """
            SELECT *
            FROM training_runs
            ORDER BY training_time DESC, id DESC
            LIMIT 1
            """
        )
        last_meta = cur.fetchone()

        if last_meta:
            last_batch_id = last_meta["last_seen_batch_id"]
            prev_mean = last_meta["mean_price"]
            prev_var = last_meta["var_price"]
        else:
            last_batch_id = None
            prev_mean = None
            prev_var = None

        # Calcular new_rows / new_ratio
        if last_batch_id is None:
            new_rows = total_rows
        else:
            cur.execute(
                """
                SELECT COUNT(*) AS new_rows
                FROM clean_house_prices
                WHERE price IS NOT NULL AND batch_id > %s
                """,
                (last_batch_id,),
            )
            new_rows = cur.fetchone()["new_rows"]

    new_ratio = float(new_rows) / float(total_rows) if total_rows > 0 else 0.0

    # Drift simple
    if prev_mean is not None and prev_var is not None:
        drift_mean = abs(mean_price - prev_mean) / (abs(prev_mean) + 1e-9)
        drift_var = abs(var_price - prev_var) / (abs(prev_var) + 1e-9)
        drift_score = max(drift_mean, drift_var)
    else:
        # Primera vez: sin histórico
        drift_mean = 0.0
        drift_var = 0.0
        drift_score = 0.0

    print(f"[train_house_price_model] new_rows={new_rows}, new_ratio={new_ratio:.4f}, "
          f"drift_mean={drift_mean:.4f}, drift_var={drift_var:.4f}, drift_score={drift_score:.4f}")

    # Política sencilla de retraining
    NEW_RATIO_THRESHOLD = 0.05  # 5% de nuevas filas
    DRIFT_THRESHOLD = 0.10      # 10% de cambio relativo

    if last_meta is None:
        retrain_decision = "trained_first_time"
        retrain_reason = "No previous training_runs; first training."
        proceed_training = True
    else:
        if (new_ratio >= NEW_RATIO_THRESHOLD) or (drift_score >= DRIFT_THRESHOLD):
            retrain_decision = "trained"
            retrain_reason = (
                f"new_ratio={new_ratio:.4f} (>= {NEW_RATIO_THRESHOLD}) or "
                f"drift_score={drift_score:.4f} (>= {DRIFT_THRESHOLD})"
            )
            proceed_training = True
        else:
            retrain_decision = "skipped"
            retrain_reason = (
                f"new_ratio={new_ratio:.4f} < {NEW_RATIO_THRESHOLD} AND "
                f"drift_score={drift_score:.4f} < {DRIFT_THRESHOLD}"
            )
            proceed_training = False

    print(f"[train_house_price_model] Retrain decision: {retrain_decision} | {retrain_reason}")

    # -------------------- 1) MLflow setup --------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment_name = f"group_{GROUP_NUMBER}_house_prices"
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()

    # Si NO vamos a re-entrenar, registramos solo la decisión en MLflow y en training_runs
    if not proceed_training:
        with mlflow.start_run(run_name="house_price_random_forest_skipped") as run:
            mlflow.set_tag("retrain_decision", retrain_decision)
            mlflow.log_param("retrain_reason", retrain_reason)
            mlflow.log_metric("new_ratio", new_ratio)
            mlflow.log_metric("drift_score", drift_score)
            if last_meta is not None:
                mlflow.log_param("last_seen_batch_id", last_meta["last_seen_batch_id"])

            run_id = run.info.run_id

        # Escribimos en training_runs
        conn = _get_mysql_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO training_runs (
                      run_id,
                      last_seen_batch_id,
                      total_rows,
                      new_rows,
                      new_ratio,
                      mean_price,
                      var_price,
                      drift_mean,
                      drift_var,
                      drift_score,
                      retrain_decision,
                      promotion_decision,
                      decision_reason,
                      promoted_to_production
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        run_id,
                        last_meta["last_seen_batch_id"] if last_meta else None,
                        total_rows,
                        new_rows,
                        new_ratio,
                        mean_price,
                        var_price,
                        drift_mean,
                        drift_var,
                        drift_score,
                        retrain_decision,
                        "none",
                        retrain_reason,
                        0,
                    ),
                )
            conn.commit()
        finally:
            conn.close()

        print("[train_house_price_model] Retrain skipped. Exiting task.")
        return

    # -------------------- 2) Cargar datos y entrenar --------------------
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
                price,
                batch_id
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

    # Convertir a float
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

    X = df[feature_cols].to_numpy(dtype="float64")
    y = df[target_col].to_numpy(dtype="float64")
    # último batch_id observado
    last_seen_batch_id = int(df["batch_id"].max()) if "batch_id" in df.columns else None

    print(f"[train_house_price_model] Shape of X: {X.shape}, y: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name="house_price_random_forest") as run:
        params = {
            "n_estimators": 200,
            "max_depth": 10,
            "random_state": 42,
        }
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_param("feature_cols", ",".join(feature_cols))
        mlflow.log_metric("new_ratio", new_ratio)
        mlflow.log_metric("drift_score", drift_score)
        mlflow.set_tag("retrain_decision", retrain_decision)
        mlflow.log_param("retrain_reason", retrain_reason)
        if last_seen_batch_id is not None:
            mlflow.log_param("last_seen_batch_id", last_seen_batch_id)

        # Registrar modelo en MLflow Model Registry
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        mlflow.set_tag("source", "airflow_dag_house_prices_train_model")

        run_id = run.info.run_id
        print(f"Run ID: {run_id}")
        print(f"RMSE: {rmse:.2f}, R2: {r2:.4f}")

        # -------------------- 3) Política de promoción a Production --------------------
        # Encontrar la versión creada para este run
        versions = client.search_model_versions(
            f"name='{REGISTERED_MODEL_NAME}' and run_id='{run_id}'"
        )
        if not versions:
            raise RuntimeError("Could not find model version for this run in Model Registry.")
        this_version = versions[0]
        this_version_num = int(this_version.version)

        # Leer modelo actual en Production (si existe)
        latest_prod = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
        if not latest_prod:
            # No hay modelo en Production -> promovemos directo
            promote = True
            promotion_reason = "No production model exists yet; promoting first good run."
            rmse_prod = None
            r2_prod = None
        else:
            prod_ver = latest_prod[0]
            prod_run = client.get_run(prod_ver.run_id)
            rmse_prod = float(prod_run.data.metrics.get("rmse", float("inf")))
            r2_prod = float(prod_run.data.metrics.get("r2", float("-1e9")))
            promote = (rmse <= rmse_prod) and (r2 >= r2_prod)
            promotion_reason = (
                f"rmse_new={rmse:.4f} vs rmse_prod={rmse_prod:.4f}, "
                f"r2_new={r2:.4f} vs r2_prod={r2_prod:.4f}. "
                "Promote if rmse_new <= rmse_prod and r2_new >= r2_prod."
            )

        if promote:
            client.transition_model_version_stage(
                name=REGISTERED_MODEL_NAME,
                version=this_version_num,
                stage="Production",
                archive_existing_versions=True,
            )
            promotion_decision = "promoted"
            promoted_to_production = 1
            print(f"[train_house_price_model] Model version {this_version_num} promoted to Production.")
        else:
            promotion_decision = "rejected"
            promoted_to_production = 0
            print(f"[train_house_price_model] Model version {this_version_num} NOT promoted to Production.")

        mlflow.set_tag("promotion_decision", promotion_decision)
        mlflow.log_param("promotion_reason", promotion_reason)

    # -------------------- 4) Escribir training_runs --------------------
    conn = _get_mysql_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO training_runs (
                  run_id,
                  last_seen_batch_id,
                  total_rows,
                  new_rows,
                  new_ratio,
                  mean_price,
                  var_price,
                  drift_mean,
                  drift_var,
                  drift_score,
                  retrain_decision,
                  promotion_decision,
                  decision_reason,
                  promoted_to_production
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    run_id,
                    last_seen_batch_id,
                    total_rows,
                    new_rows,
                    new_ratio,
                    mean_price,
                    var_price,
                    drift_mean,
                    drift_var,
                    drift_score,
                    retrain_decision,
                    promotion_decision,
                    promotion_reason,
                    promoted_to_production,
                ),
            )
        conn.commit()
    finally:
        conn.close()

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
