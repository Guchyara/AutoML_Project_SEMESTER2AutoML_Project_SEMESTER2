import logging
import os
import sys
from datetime import timedelta
from pathlib import Path

import mlflow
import pandas as pd
import requests
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

SCRIPTS_DIR = "/opt/airflow/scripts"
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from model_load_predict_operator import ModelLoadAndPredictOperator


MODEL_NAME = os.getenv("INFERENCE_MODEL_NAME", "titanic-survival-predictor")
API_URL = os.getenv("INFERENCE_API_URL", "http://inference-api:8000")
INFERENCE_EXPERIMENT_NAME = os.getenv("INFERENCE_EXPERIMENT_NAME", "mlops_inference")
DATA_ROOT = Path("/opt/airflow/data")
FEATURE_COLUMNS = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "IsAlone"]


@dag(
    dag_id="mlops_inference_pipeline",
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
    default_args={"owner": "airflow", "retries": 1, "retry_delay": timedelta(minutes=1)},
    tags=["mlops", "inference", "production"],
    description="Batch inference through MLflow model and FastAPI service",
)
def mlops_inference_pipeline():
    @task
    def prepare_batch():
        source = DATA_ROOT / "model_split" / "kaggle_test.csv"
        output = DATA_ROOT / "inference" / "batch.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        dataframe = pd.read_csv(source)
        dataframe[FEATURE_COLUMNS].to_csv(output, index=False)
        return str(output)

    @task
    def check_api_health():
        response = requests.get(f"{API_URL}/health", timeout=30)
        response.raise_for_status()
        health = response.json()
        if not health.get("model_loaded"):
            raise RuntimeError(f"Inference API is unhealthy: {health}")
        return health

    @task
    def run_api_batch(batch_path):
        import sqlite3

        dataframe = pd.read_csv(batch_path)
        payload = {"data": dataframe.values.tolist(), "feature_names": dataframe.columns.tolist()}
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=60)
        response.raise_for_status()
        body = response.json()
        output = DATA_ROOT / "inference" / "api_results.csv"
        result = dataframe.copy()
        result["prediction"] = body["predictions"]
        result["model_name"] = body["model_name"]
        result["model_stage"] = body["model_stage"]
        result.to_csv(output, index=False)
        database_path = DATA_ROOT / "inference" / "inference_results.db"
        with sqlite3.connect(database_path) as connection:
            result.to_sql("predictions", connection, if_exists="replace", index=False)
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        mlflow.set_experiment(INFERENCE_EXPERIMENT_NAME)
        with mlflow.start_run(run_name="api_batch_inference"):
            mlflow.log_params({"model_name": body["model_name"], "model_stage": body["model_stage"], "rows": len(result)})
            mlflow.log_metric("prediction_rows", len(result))
            mlflow.log_artifact(str(output), artifact_path="inference")
        logging.info("API batch inference saved to %s", output)
        logging.info("Inference results persisted to %s", database_path)
        return str(output)

    batch = prepare_batch()
    health = check_api_health()
    api_result = run_api_batch(batch)
    health >> api_result
    operator_result = ModelLoadAndPredictOperator(
        task_id="mlflow_model_batch_prediction",
        input_path="/opt/airflow/data/inference/batch.csv",
        output_path="/opt/airflow/data/inference/operator_results.csv",
        model_name=MODEL_NAME,
        model_stage="Production",
    )
    batch >> operator_result


mlops_inference_pipeline()

