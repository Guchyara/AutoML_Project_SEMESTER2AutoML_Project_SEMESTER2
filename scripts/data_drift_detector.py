import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd


class DataDriftMonitor:
    """Dependency-light numeric drift monitor using standardized mean shifts."""

    def __init__(self, threshold=0.2):
        self.threshold = threshold
        self.reference_data = None

    def set_reference_data(self, reference_df):
        self.reference_data = reference_df.select_dtypes(include="number").copy().fillna(0)

    def check_data_drift(self, current_df):
        if self.reference_data is None:
            raise ValueError("Reference data must be set before drift check")
        current = current_df.reindex(columns=self.reference_data.columns).select_dtypes(include="number").fillna(0)
        feature_scores = {}
        for column in self.reference_data.columns:
            reference_std = float(self.reference_data[column].std()) or 1.0
            score = abs(float(current[column].mean()) - float(self.reference_data[column].mean())) / reference_std
            feature_scores[column] = score
        drifted = [column for column, score in feature_scores.items() if score > self.threshold]
        overall_score = max(feature_scores.values(), default=0.0)
        return {
            "drift_detected": bool(drifted),
            "drift_score": float(overall_score),
            "features_drifted": len(drifted),
            "share_drifted_features": len(drifted) / max(len(feature_scores), 1),
            "feature_scores": feature_scores,
            "reference_rows": len(self.reference_data),
            "current_rows": len(current),
            "threshold": self.threshold,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def log_to_mlflow(self, drift_metrics, dataset_name, experiment_name="mlops_monitoring"):
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"data_drift_{dataset_name}"):
            mlflow.log_params({"dataset": dataset_name, "drift_detected": drift_metrics["drift_detected"], "threshold": drift_metrics["threshold"]})
            mlflow.log_metrics({
                "drift_score": drift_metrics["drift_score"],
                "features_drifted": drift_metrics["features_drifted"],
                "share_drifted_features": drift_metrics["share_drifted_features"],
            })
            mlflow.log_dict(drift_metrics, "drift_report.json")

