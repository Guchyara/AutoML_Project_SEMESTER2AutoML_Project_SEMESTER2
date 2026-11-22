import logging
import os
from datetime import datetime, timezone

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "IsAlone"]


class ModelRetrainer:
    def __init__(self, tracking_uri=None):
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def retrain_if_needed(self, training_data_path, model_name, drift_metrics):
        if not drift_metrics.get("drift_detected"):
            logging.info("Drift not detected; retraining skipped")
            return {"status": "skipped", "reason": "no_drift"}
        dataframe = pd.read_csv(training_data_path)
        X = dataframe[FEATURE_COLUMNS]
        y = dataframe["Survived"]
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        with mlflow.start_run(run_name=f"retraining_{model_name}") as run:
            model.fit(X_train, y_train)
            score = f1_score(y_valid, model.predict(X_valid), average="weighted")
            mlflow.log_metric("f1_score", float(score))
            mlflow.log_params({"model_name": model_name, "reason": "data_drift", "retraining_timestamp": datetime.now(timezone.utc).isoformat()})
            mlflow.sklearn.log_model(model, "model")
            version = self.client.create_model_version(name=model_name, source=f"runs:/{run.info.run_id}/model", run_id=run.info.run_id)
            self.client.transition_model_version_stage(name=model_name, version=version.version, stage="Production", archive_existing_versions=True)
        return {"status": "retrained", "version": int(version.version), "f1_score": float(score)}

