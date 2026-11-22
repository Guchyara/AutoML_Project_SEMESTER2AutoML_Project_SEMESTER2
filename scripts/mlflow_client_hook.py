import os

import mlflow
from airflow.hooks.base import BaseHook
from mlflow.tracking import MlflowClient


class MLflowClientHook(BaseHook):
    """Small Airflow hook around MlflowClient for dynamic DAG tasks."""

    def __init__(self, tracking_uri=None):
        super().__init__()
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        self._client = None

    def get_conn(self):
        if self._client is None:
            mlflow.set_tracking_uri(self.tracking_uri)
            self._client = MlflowClient(tracking_uri=self.tracking_uri)
        return self._client

    def get_or_create_experiment(self, experiment_name):
        client = self.get_conn()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = client.create_experiment(experiment_name)
            experiment = client.get_experiment(experiment_id)
        return experiment

    def register_best_model(self, model_name, model_uri, run_id):
        client = self.get_conn()
        try:
            client.get_registered_model(model_name)
        except Exception:
            client.create_registered_model(model_name)
        version = client.create_model_version(name=model_name, source=model_uri, run_id=run_id)
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Production",
            archive_existing_versions=True,
        )
        return int(version.version)

