import json
import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score
from sklearn.model_selection import train_test_split


class DynamicAutoMLTrainer:
    def _models(self, task_type):
        if task_type == "classification":
            return {
                "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
                "random_forest": RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
                "gradient_boosting": GradientBoostingClassifier(random_state=42),
            }
        return {
            "linear_regression": LinearRegression(),
            "random_forest": RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
            "gradient_boosting": GradientBoostingRegressor(random_state=42),
        }

    @staticmethod
    def _metrics(task_type, y_true, y_pred):
        if task_type == "classification":
            return {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
                "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            }
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    def train_dataset(self, dataframe, dataset_name, target_column, task_type, metric, experiment_name, output_path):
        X = dataframe.drop(columns=[target_column])
        y = dataframe[target_column]
        stratify = y if task_type == "classification" and y.value_counts().min() >= 2 else None
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
        available_metrics = {"accuracy", "f1", "precision", "recall"} if task_type == "classification" else {"rmse", "mae", "r2"}
        selection_metric = metric if metric in available_metrics else ("f1" if task_type == "classification" else "rmse")
        if metric not in available_metrics:
            logging.warning("Metric %s не подходит для %s; используется %s", metric, task_type, selection_metric)
        mlflow.set_experiment(experiment_name)
        results = {}
        with mlflow.start_run(run_name=f"dynamic_{dataset_name}") as parent_run:
            mlflow.log_params({"dataset": dataset_name, "task_type": task_type, "target_column": target_column, "selection_metric": selection_metric})
            for model_name, model in self._models(task_type).items():
                with mlflow.start_run(run_name=model_name, nested=True) as model_run:
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_valid)
                    metrics = self._metrics(task_type, y_valid, predictions)
                    mlflow.log_params({"model_type": model_name, "features": X.shape[1]})
                    mlflow.log_metrics(metrics)
                    mlflow.sklearn.log_model(model, f"models/{model_name}")
                    results[model_name] = {"metrics": metrics, "run_id": model_run.info.run_id}

            if task_type == "regression":
                best_name = min(results, key=lambda name: results[name]["metrics"].get(selection_metric, results[name]["metrics"]["rmse"]))
            else:
                best_name = max(results, key=lambda name: results[name]["metrics"].get(selection_metric, results[name]["metrics"]["f1"]))
            comparison = {
                "dataset": dataset_name,
                "task_type": task_type,
                "target_column": target_column,
                "selection_metric": selection_metric,
                "best_model": best_name,
                "parent_run_id": parent_run.info.run_id,
                "models": results,
            }
            mlflow.log_dict(comparison, "dynamic_comparison.json")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        logging.info("Dataset %s: best model=%s", dataset_name, best_name)
        return comparison

