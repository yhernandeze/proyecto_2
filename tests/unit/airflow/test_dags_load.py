# tests/unit/airflow/test_dags_load.py
import os
from airflow.models import DagBag

DAGS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dags")


def _load_dagbag():
    return DagBag(dag_folder=DAGS_FOLDER, include_examples=False)


def test_dags_import_ok():
    dag_bag = _load_dagbag()
    assert len(dag_bag.import_errors) == 0, f"Import errors: {dag_bag.import_errors}"


def test_house_prices_ingest_raw_dag_exists():
    dag_bag = _load_dagbag()
    dag = dag_bag.get_dag("house_prices_ingest_raw")
    assert dag is not None
    assert len(dag.tasks) == 2
    task_ids = {t.task_id for t in dag.tasks}
    assert task_ids == {"fetch_batch", "insert_raw"}


def test_house_prices_build_clean_dag_exists():
    dag_bag = _load_dagbag()
    dag = dag_bag.get_dag("house_prices_build_clean")
    assert dag is not None
    assert len(dag.tasks) == 1
    assert {t.task_id for t in dag.tasks} == {"build_clean_from_latest_batch"}


def test_house_prices_train_model_dag_exists():
    dag_bag = _load_dagbag()
    dag = dag_bag.get_dag("house_prices_train_model")
    assert dag is not None
    assert len(dag.tasks) == 1
    assert {t.task_id for t in dag.tasks} == {"train_house_model"}
