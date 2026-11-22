import logging
import os
import sys
from datetime import timedelta

import mlflow
from mlflow.tracking import MlflowClient
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

SCRIPTS_DIR = "/opt/airflow/scripts"
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from data_loader import (
    DATA_SOURCE,
    download_titanic_data,
    load_data_from_local,
    save_data_locally,
)
from data_preprocessor import preprocess_titanic_data


MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "titanic_basic_pipeline",
)
FINAL_DIR = "/opt/airflow/data/final"

mlflow.set_tracking_uri(MLFLOW_URI)


def _configure_mlflow():
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    experiment_name = MLFLOW_EXPERIMENT
    if experiment and experiment.lifecycle_stage == "deleted":
        experiment_name = f"{MLFLOW_EXPERIMENT}_v2"
        logging.warning(
            "Эксперимент %s помечен deleted; используется %s",
            MLFLOW_EXPERIMENT,
            experiment_name,
        )
    mlflow.set_experiment(experiment_name)
    return experiment_name


def download_and_log_data():
    train_df, test_df = download_titanic_data()
    loaded_dir = "/opt/airflow/data/loaded"
    save_data_locally(train_df, test_df, loaded_dir)

    experiment_name = _configure_mlflow()
    with mlflow.start_run(run_name="data_download"):
        mlflow.log_param("dataset", "titanic")
        mlflow.log_param("source_mode", DATA_SOURCE)
        mlflow.log_param("experiment_name", experiment_name)
        mlflow.log_metric("train_samples", len(train_df))
        mlflow.log_metric("test_samples", len(test_df))
        mlflow.log_metric("total_features", train_df.shape[1])

    logging.info("Загрузка данных завершена: train=%s, test=%s", train_df.shape, test_df.shape)


def preprocess_and_log_data():
    train_df, test_df = load_data_from_local("/opt/airflow/data/loaded")
    processed_train, processed_test, _ = preprocess_titanic_data(train_df, test_df)
    save_data_locally(processed_train, processed_test, FINAL_DIR)

    experiment_name = _configure_mlflow()
    with mlflow.start_run(run_name="data_preprocessing"):
        mlflow.log_param("missing_value_strategy", "median/mode")
        mlflow.log_param("categorical_encoding", "label_encoding")
        mlflow.log_param("experiment_name", experiment_name)
        mlflow.log_metric("processed_train_samples", len(processed_train))
        mlflow.log_metric("processed_test_samples", len(processed_test))
        mlflow.log_metric("processed_features", processed_train.shape[1])
        mlflow.log_text(
            processed_train.head(10).to_csv(index=False),
            "processed_data_sample.csv",
        )

    logging.info(
        "Предобработка завершена: train=%s, test=%s",
        processed_train.shape,
        processed_test.shape,
    )


def log_dataset_summary():
    train_df, test_df = load_data_from_local(FINAL_DIR)
    summary = {
        "total_samples": len(train_df) + len(test_df),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "features_count": train_df.shape[1],
        "memory_usage_mb": train_df.memory_usage(deep=True).sum() / 1024**2,
    }

    experiment_name = _configure_mlflow()
    with mlflow.start_run(run_name="dataset_summary"):
        mlflow.log_param("dataset", "titanic")
        mlflow.log_param("source_mode", DATA_SOURCE)
        mlflow.log_param("experiment_name", experiment_name)
        mlflow.log_param("features_list", str(list(train_df.columns)))
        for key, value in summary.items():
            mlflow.log_metric(key, value)

    logging.info("Итоговая сводка: %s", summary)


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="titanic_basic_pipeline",
    default_args=default_args,
    description="Базовый пайплайн Titanic",
    schedule_interval=None,
    catchup=False,
    tags=["titanic", "mlflow", "kaggle"],
) as dag:
    download_task = PythonOperator(
        task_id="download_titanic_data",
        python_callable=download_and_log_data,
    )
    preprocess_task = PythonOperator(
        task_id="preprocess_titanic_data",
        python_callable=preprocess_and_log_data,
    )
    summary_task = PythonOperator(
        task_id="log_dataset_summary",
        python_callable=log_dataset_summary,
    )

    download_task >> preprocess_task >> summary_task

