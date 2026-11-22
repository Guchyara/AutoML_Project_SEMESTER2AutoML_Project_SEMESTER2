import json
import logging
import os
import sys
from datetime import timedelta
from pathlib import Path

import mlflow
from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.utils.dates import days_ago

SCRIPTS_DIR = "/opt/airflow/scripts"
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from automl_trainer import DynamicAutoMLTrainer
from data_manager import DynamicDataManager
from mlflow_client_hook import MLflowClientHook
from task_detector import TaskDetector


MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_PREFIX = os.getenv("DYNAMIC_EXPERIMENT_PREFIX", "dynamic_ml_pipeline")
REGISTERED_MODEL_PREFIX = os.getenv("DYNAMIC_REGISTERED_MODEL_PREFIX", "dynamic")
DATA_ROOT = Path("/opt/airflow/data")


@dag(
    dag_id="dynamic_ml_pipeline",
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
    default_args={"owner": "airflow", "retries": 1, "retry_delay": timedelta(minutes=1)},
    tags=["dynamic", "automl", "mlflow", "datasets"],
    description="Динамический ML-пайплайн для нескольких датасетов из Airflow Variable",
)
def dynamic_ml_pipeline():
    @task
    def get_dataset_names():
        configured = Variable.get("dynamic_datasets", default_var='["titanic", "iris"]')
        if isinstance(configured, str):
            try:
                names = json.loads(configured)
            except json.JSONDecodeError:
                names = [item.strip() for item in configured.split(",") if item.strip()]
        else:
            names = configured
        if not isinstance(names, list) or not names:
            raise ValueError("Airflow Variable dynamic_datasets должна быть непустым JSON-списком")
        names = [str(name).strip() for name in names]
        logging.info("Датасеты из Airflow Variable dynamic_datasets: %s", names)
        return names

    @task
    def download_dataset(dataset_name):
        return DynamicDataManager().download_dataset(dataset_name)

    @task
    def preprocess_dataset(dataset_name):
        manager = DynamicDataManager()
        raw_path = manager.raw_root / dataset_name / "raw.csv"
        dataframe = manager.preprocess_data(__import__("pandas").read_csv(raw_path), dataset_name)
        output_path = manager.processed_root / dataset_name / "processed.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(output_path, index=False)
        logging.info("Датасет %s предобработан: %s", dataset_name, dataframe.shape)
        return str(output_path)

    @task
    def train_dataset(dataset_name):
        import pandas as pd

        manager = DynamicDataManager()
        config = manager.get_config(dataset_name)
        processed_path = manager.processed_root / dataset_name / "processed.csv"
        dataframe = pd.read_csv(processed_path)
        task_type = TaskDetector().detect_task_type(dataframe[config["target_column"]])
        experiment_name = f"{EXPERIMENT_PREFIX}_{dataset_name}"
        output_path = DATA_ROOT / "dynamic_results" / dataset_name / "comparison.json"
        comparison = DynamicAutoMLTrainer().train_dataset(
            dataframe,
            dataset_name,
            config["target_column"],
            task_type,
            config["evaluation_metric"],
            experiment_name,
            output_path,
        )
        comparison["experiment_name"] = experiment_name
        return comparison

    @task
    def register_and_summarize(comparison):
        hook = MLflowClientHook(MLFLOW_URI)
        best_model = comparison["best_model"]
        best_run_id = comparison["models"][best_model]["run_id"]
        registered_name = f"{REGISTERED_MODEL_PREFIX}-{comparison['dataset']}-predictor"
        version = hook.register_best_model(
            registered_name,
            f"runs:/{best_run_id}/models/{best_model}",
            best_run_id,
        )
        result = {
            "dataset": comparison["dataset"],
            "task_type": comparison["task_type"],
            "experiment_name": comparison["experiment_name"],
            "best_model": best_model,
            "selection_metric": comparison["selection_metric"],
            "best_metric_value": comparison["models"][best_model]["metrics"][comparison["selection_metric"]],
            "registered_model": registered_name,
            "model_version": version,
        }
        logging.info("Зарегистрирована %s version %s", registered_name, version)
        return result

    @task
    def final_report(results):
        results = list(results)
        report_path = DATA_ROOT / "dynamic_results" / "dynamic_pipeline_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        hook = MLflowClientHook(MLFLOW_URI)
        experiment = hook.get_or_create_experiment(f"{EXPERIMENT_PREFIX}_summary")
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(experiment.name)
        with mlflow.start_run(run_name="dynamic_pipeline_summary"):
            mlflow.log_metric("datasets_processed", len(results))
            mlflow.log_dict({"results": results}, "dynamic_pipeline_report.json")
        logging.info("Итоговый dynamic report: %s", report_path)
        return str(report_path)

    names = get_dataset_names()
    downloaded = download_dataset.expand(dataset_name=names)
    processed = preprocess_dataset.expand(dataset_name=names)
    downloaded >> processed
    comparisons = train_dataset.expand(dataset_name=names)
    processed >> comparisons
    registered = register_and_summarize.expand(comparison=comparisons)
    final_report(registered)


dynamic_ml_pipeline()

