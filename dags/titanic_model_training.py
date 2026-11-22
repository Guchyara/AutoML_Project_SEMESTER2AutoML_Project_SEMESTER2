import json
import logging
import os
import sys
from datetime import timedelta
from pathlib import Path

import mlflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split

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
from model_trainer import ModelTrainer


MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
BASE_EXPERIMENT_NAME = os.getenv(
    "MODEL_EXPERIMENT_NAME",
    "titanic_model_comparison",
)
REGISTERED_MODEL_NAME = os.getenv(
    "REGISTERED_MODEL_NAME",
    "titanic-survival-predictor",
)

DATA_ROOT = Path("/opt/airflow/data")
RAW_DIR = DATA_ROOT / "model_raw"
PROCESSED_DIR = DATA_ROOT / "model_processed"
SPLIT_DIR = DATA_ROOT / "model_split"
RESULTS_DIR = DATA_ROOT / "model_results"

mlflow.set_tracking_uri(MLFLOW_URI)


def configure_mlflow():
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()
    existing = client.get_experiment_by_name(BASE_EXPERIMENT_NAME)
    experiment_name = BASE_EXPERIMENT_NAME
    if existing and existing.lifecycle_stage == "deleted":
        experiment_name = f"{BASE_EXPERIMENT_NAME}_v2"
        logging.warning(
            "Эксперимент %s удалён; используется %s",
            BASE_EXPERIMENT_NAME,
            experiment_name,
        )
    mlflow.set_experiment(experiment_name)
    return experiment_name


def download_data():
    train_df, test_df = download_titanic_data()
    save_data_locally(train_df, test_df, RAW_DIR)

    experiment_name = configure_mlflow()
    with mlflow.start_run(run_name="data_download"):
        mlflow.log_param("dataset", "titanic")
        mlflow.log_param("source_mode", DATA_SOURCE)
        mlflow.log_param("experiment_name", experiment_name)
        mlflow.log_metric("train_samples", len(train_df))
        mlflow.log_metric("test_samples", len(test_df))

    logging.info("Model comparison: data loaded train=%s test=%s", train_df.shape, test_df.shape)


def preprocess_data():
    train_df, test_df = load_data_from_local(RAW_DIR)
    processed_train, processed_test, _ = preprocess_titanic_data(train_df, test_df)
    save_data_locally(processed_train, processed_test, PROCESSED_DIR)

    experiment_name = configure_mlflow()
    with mlflow.start_run(run_name="data_preprocessing"):
        mlflow.log_param("experiment_name", experiment_name)
        mlflow.log_param("missing_value_strategy", "median/mode")
        mlflow.log_param("categorical_encoding", "label_encoding")
        mlflow.log_metric("processed_train_samples", len(processed_train))
        mlflow.log_metric("processed_test_samples", len(processed_test))
        mlflow.log_metric("processed_features", processed_train.shape[1])

    logging.info(
        "Model comparison: preprocessing completed train=%s test=%s",
        processed_train.shape,
        processed_test.shape,
    )


def split_data():
    train_df, test_df = load_data_from_local(PROCESSED_DIR)
    train_part, validation_part = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,
        stratify=train_df["Survived"],
    )
    save_data_locally(train_part, validation_part, SPLIT_DIR)
    test_df.to_csv(SPLIT_DIR / "kaggle_test.csv", index=False)

    experiment_name = configure_mlflow()
    with mlflow.start_run(run_name="data_split"):
        mlflow.log_param("experiment_name", experiment_name)
        mlflow.log_param("split_strategy", "stratified_train_test_split")
        mlflow.log_metric("training_samples", len(train_part))
        mlflow.log_metric("validation_samples", len(validation_part))
        mlflow.log_metric("holdout_test_samples", len(test_df))

    logging.info(
        "Model comparison: split train=%s validation=%s",
        train_part.shape,
        validation_part.shape,
    )


def train_and_compare_models():
    train_df, validation_df = load_data_from_local(SPLIT_DIR)
    experiment_name = configure_mlflow()

    with mlflow.start_run(run_name="model_training_comparison"):
        mlflow.log_param("experiment_name", experiment_name)
        mlflow.log_param("selection_metric", "f1_score")
        trainer = ModelTrainer(random_state=42)
        results = trainer.train_models(
            train_df,
            validation_df,
            experiment_name,
        )
        best_model_name, best_model_info = trainer.find_best_model(
            results,
            metric="f1_score",
        )
        comparison = trainer.save_comparison(
            results,
            best_model_name,
            RESULTS_DIR / "model_comparison.json",
        )

        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metric("best_model_f1", best_model_info["metrics"]["f1_score"])
        mlflow.log_metric(
            "best_model_accuracy",
            best_model_info["metrics"]["accuracy"],
        )
        mlflow.log_dict(comparison, "model_comparison.json")

    logging.info(
        "Best model: %s with f1_score=%.4f",
        best_model_name,
        best_model_info["metrics"]["f1_score"],
    )
    return best_model_name


def register_best_model():
    comparison_path = RESULTS_DIR / "model_comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    best_model_name = comparison["best_model"]
    best_model_run_id = comparison["models"][best_model_name]["run_id"]

    configure_mlflow()
    client = MlflowClient()
    try:
        client.get_registered_model(REGISTERED_MODEL_NAME)
    except Exception:
        client.create_registered_model(REGISTERED_MODEL_NAME)

    model_uri = f"runs:/{best_model_run_id}/{best_model_name}"
    model_version = client.create_model_version(
        name=REGISTERED_MODEL_NAME,
        source=model_uri,
        run_id=best_model_run_id,
    )
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=model_version.version,
        stage="Production",
        archive_existing_versions=True,
    )

    with mlflow.start_run(run_name="model_registration"):
        mlflow.log_param("registered_model", REGISTERED_MODEL_NAME)
        mlflow.log_param("model_version", str(model_version.version))
        mlflow.log_param("best_model_type", best_model_name)
        mlflow.log_metric("model_version_number", int(model_version.version))

    logging.info(
        "Registered %s version %s in Production",
        REGISTERED_MODEL_NAME,
        model_version.version,
    )


def create_comparison_report():
    comparison_path = RESULTS_DIR / "model_comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    experiment_name = configure_mlflow()

    report_lines = [
        "TITANIC MODEL COMPARISON REPORT",
        "================================",
        f"Best model by {comparison['selection_metric']}: {comparison['best_model']}",
        "",
        "MODEL METRICS:",
    ]
    for model_name, result in comparison["models"].items():
        metrics = result["metrics"]
        report_lines.extend(
            [
                "",
                model_name.upper(),
                f"  Accuracy:  {metrics['accuracy']:.4f}",
                f"  Precision: {metrics['precision']:.4f}",
                f"  Recall:    {metrics['recall']:.4f}",
                f"  F1-score:  {metrics['f1_score']:.4f}",
                f"  Run ID:    {result['run_id']}",
            ]
        )

    with mlflow.start_run(run_name="comparison_report"):
        mlflow.log_param("experiment_name", experiment_name)
        mlflow.log_param("best_model", comparison["best_model"])
        mlflow.log_dict(comparison, "final_comparison_report.json")
        mlflow.log_text("\n".join(report_lines), "model_comparison_summary.txt")

    logging.info("Comparison report logged for %s", comparison["best_model"])


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
}

with DAG(
    dag_id="titanic_model_comparison",
    default_args=default_args,
    description="Сравнение моделей Titanic с MLflow Registry",
    schedule_interval=None,
    catchup=False,
    tags=["titanic", "mlflow", "model-comparison", "registry"],
) as dag:
    download_task = PythonOperator(
        task_id="download_titanic_data",
        python_callable=download_data,
    )
    preprocess_task = PythonOperator(
        task_id="preprocess_titanic_data",
        python_callable=preprocess_data,
    )
    split_task = PythonOperator(
        task_id="split_train_validation_data",
        python_callable=split_data,
    )
    train_task = PythonOperator(
        task_id="train_and_compare_models",
        python_callable=train_and_compare_models,
    )
    register_task = PythonOperator(
        task_id="register_best_model",
        python_callable=register_best_model,
    )
    report_task = PythonOperator(
        task_id="create_comparison_report",
        python_callable=create_comparison_report,
    )

    download_task >> preprocess_task >> split_task >> train_task
    train_task >> [register_task, report_task]

