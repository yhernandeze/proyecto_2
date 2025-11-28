import os

import pytest

try:
    from airflow.models import DagBag  # type: ignore
except ImportError:
    pytest.skip("Airflow not installed in this environment", allow_module_level=True)

from shared.feature_contract import FEATURE_COLUMNS

DAGS_FOLDER = os.path.join(os.path.dirname(__file__), "..", "airflow", "dags")


def _load_dagbag():
    return DagBag(dag_folder=DAGS_FOLDER, include_examples=False)


def test_dags_import_ok():
    dag_bag = _load_dagbag()
    assert len(dag_bag.import_errors) == 0, f"Import errors: {dag_bag.import_errors}"


def test_house_prices_dags_exist():
    dag_bag = _load_dagbag()
    assert dag_bag.get_dag("house_prices_ingest_raw")
    assert dag_bag.get_dag("house_prices_build_clean")
    assert dag_bag.get_dag("house_prices_train_model")


def test_training_feature_contract():
    dag_bag = _load_dagbag()
    dag = dag_bag.get_dag("house_prices_train_model")
    task = dag.get_task("train_house_model")
    module = task.python_callable.__module__
    mod = __import__(module, fromlist=["FEATURE_COLS"])
    assert getattr(mod, "FEATURE_COLS") == FEATURE_COLUMNS
