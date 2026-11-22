import os
import sqlite3
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
from airflow.models.baseoperator import BaseOperator


class ModelLoadAndPredictOperator(BaseOperator):
    """Load a registered MLflow model and run a batch prediction."""

    template_fields = ("input_path", "output_path", "model_name", "model_stage")

    def __init__(self, *, input_path, output_path, model_name, model_stage="Production", **kwargs):
        super().__init__(**kwargs)
        self.input_path = input_path
        self.output_path = output_path
        self.model_name = model_name
        self.model_stage = model_stage

    def execute(self, context):
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(os.getenv("INFERENCE_EXPERIMENT_NAME", "mlops_inference"))
        model_uri = f"models:/{self.model_name}/{self.model_stage}"
        model = mlflow.pyfunc.load_model(model_uri)
        dataframe = pd.read_csv(self.input_path)
        predictions = model.predict(dataframe)
        result = dataframe.copy()
        result["prediction"] = predictions
        result["model_name"] = self.model_name
        result["model_stage"] = self.model_stage
        output_path = Path(self.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        database_path = output_path.parent / "inference_results.db"
        with sqlite3.connect(database_path) as connection:
            result.to_sql("operator_predictions", connection, if_exists="replace", index=False)
        with mlflow.start_run(run_name=f"batch_inference_{self.model_name}"):
            mlflow.log_params({"model_name": self.model_name, "model_stage": self.model_stage, "input_rows": len(dataframe)})
            mlflow.log_metric("prediction_rows", len(result))
            mlflow.log_artifact(str(output_path), artifact_path="inference")
        self.log.info("Batch prediction completed: %s rows -> %s", len(result), output_path)
        return str(output_path)

