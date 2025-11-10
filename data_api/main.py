from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import pandas as pd
from sqlalchemy import create_engine, text
from ucimlrepo import fetch_ucirepo
from datetime import datetime

# ---------------------------------------------------------------------
# ENV / CONFIG
# ---------------------------------------------------------------------
MYSQL_HOST = os.getenv("MYSQL_HOST", "mysql")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "mlflow_user")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "mlflow_pass")
MYSQL_DB = os.getenv("MYSQL_DB", "datasets_db")

DB_URI = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"

BATCH_DEFAULT = 15000

app = FastAPI(
    title="Diabetes Data API",
    description="Serves diabetes dataset in random 15k batches from MySQL",
    version="1.0.0",
)

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def get_engine():
    return create_engine(DB_URI, pool_pre_ping=True, future=True)

def ensure_table():
    """
    If datasets_db.diabetes_raw does not exist or is empty, download UCI dataset
    and load it into MySQL, adding a 'served_at' column (NULL by default).
    """
    engine = get_engine()
    with engine.begin() as conn:
        # does table exist?
        exists = conn.execute(
            text("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema=:schema AND table_name='diabetes_raw'
            """),
            {"schema": MYSQL_DB}
        ).scalar_one() > 0

        need_load = False
        if not exists:
            need_load = True
        else:
            nrows = conn.execute(
                text(f"SELECT COUNT(*) FROM {MYSQL_DB}.diabetes_raw")
            ).scalar_one()
            if nrows == 0:
                need_load = True

        if not need_load:
            return  # table already populated

    # download outside transaction
    ds = fetch_ucirepo(id=296)
    X = ds.data.features.copy()
    y = None
    if hasattr(ds.data, "targets") and ds.data.targets is not None:
        y = ds.data.targets.copy()

    df = X.copy()
    if y is not None and isinstance(y, pd.DataFrame):
        for c in y.columns:
            if c not in df.columns:
                df[c] = y[c]

    # normalize colnames a bit
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    # add served_at
    df["served_at"] = pd.NaT

    # write to MySQL
    with get_engine().begin() as conn:
        df.to_sql(
            "diabetes_raw",
            con=conn,
            schema=MYSQL_DB,
            if_exists="replace",
            index=False
        )

@app.on_event("startup")
def startup_event():
    ensure_table()

# ---------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------
def health():
    return {"status": "ok"}

@app.get("/data")
def get_data(batch_size: int = BATCH_DEFAULT):
    """
    Return a random batch of rows that have not been served yet.
    Mark them as served (served_at=NOW()) so they won't appear again.
    """
    engine = get_engine()
    with engine.begin() as conn:
        # get unserved rows
        # RAND() on ~100k rows is acceptable here
        rows = conn.execute(
            text(f"""
                SELECT * FROM {MYSQL_DB}.diabetes_raw
                WHERE served_at IS NULL
                ORDER BY RAND()
                LIMIT :lim
            """),
            {"lim": batch_size}
        ).mappings().all()

        if not rows:
            return {
                "batch_number": None,
                "data": []
            }

        # collect primary identifier(s) to mark served.
        # dataset has encounter_id, so use that
        encounter_ids = [r["encounter_id"] for r in rows if "encounter_id" in r]

        if encounter_ids:
            conn.execute(
                text(f"""
                    UPDATE {MYSQL_DB}.diabetes_raw
                    SET served_at = :ts
                    WHERE encounter_id IN :ids
                """),
                {"ts": datetime.utcnow(), "ids": tuple(encounter_ids)}
            )

    # return JSON-serializable data
    return {
        "batch_number": datetime.utcnow().isoformat(),
        "data": [dict(r) for r in rows]
    }
 