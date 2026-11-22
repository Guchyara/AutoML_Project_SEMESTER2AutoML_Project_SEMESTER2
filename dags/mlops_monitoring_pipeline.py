import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

SCRIPTS_DIR = "/opt/airflow/scripts"
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from data_drift_detector import DataDriftMonitor
from model_retraining import ModelRetrainer


DATA_ROOT = Path("/opt/airflow/data")
FEATURE_COLUMNS = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "IsAlone"]


@dag(
    dag_id="mlops_monitoring_pipeline",
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
    default_args={"owner": "airflow", "retries": 1, "retry_delay": timedelta(minutes=1)},
    tags=["mlops", "monitoring", "drift"],
    description="Monitoring data drift and optional model retraining",
)
def mlops_monitoring_pipeline():
    @task
    def monitor_drift():
        reference = pd.read_csv(DATA_ROOT / "model_split" / "train.csv")[FEATURE_COLUMNS]
        current = pd.read_csv(DATA_ROOT / "inference" / "batch.csv")[FEATURE_COLUMNS]
        monitor = DataDriftMonitor(threshold=0.2)
        monitor.set_reference_data(reference)
        metrics = monitor.check_data_drift(current)
        monitor.log_to_mlflow(metrics, "titanic")
        path = DATA_ROOT / "monitoring" / "drift_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics

    @task
    def retrain_if_needed(drift_metrics):
        result = ModelRetrainer().retrain_if_needed(
            DATA_ROOT / "model_split" / "train.csv",
            os.getenv("REGISTERED_MODEL_NAME", "titanic-survival-predictor"),
            drift_metrics,
        )
        return result

    retrain_if_needed(monitor_drift())


mlops_monitoring_pipeline()

