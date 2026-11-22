import json
import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC


class ModelTrainer:
    FEATURE_COLUMNS = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "FamilySize",
        "IsAlone",
    ]

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {
            "logistic_regression": LogisticRegression(
                random_state=random_state,
                max_iter=1000,
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=200,
                random_state=random_state,
                n_jobs=-1,
            ),
            "svm": SVC(
                probability=True,
                random_state=random_state,
            ),
        }

    def prepare_features(self, train_df, validation_df):
        missing = sorted(set(self.FEATURE_COLUMNS + ["Survived"]) - set(train_df.columns))
        if missing:
            raise ValueError(
                f"В обучающем наборе отсутствуют колонки: {missing}. "
                f"Получены: {list(train_df.columns)}"
            )

        missing_validation = sorted(
            set(self.FEATURE_COLUMNS + ["Survived"]) - set(validation_df.columns)
        )
        if missing_validation:
            raise ValueError(
                f"В validation-наборе отсутствуют колонки: {missing_validation}. "
                f"Получены: {list(validation_df.columns)}"
            )

        return (
            train_df[self.FEATURE_COLUMNS],
            validation_df[self.FEATURE_COLUMNS],
            train_df["Survived"],
            validation_df["Survived"],
        )

    @staticmethod
    def _safe_params(model):
        params = {}
        for name, value in model.get_params().items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                params[name] = value
            else:
                params[name] = str(value)
        return params

    @staticmethod
    def _evaluate_model(model, X_validation, y_validation):
        predictions = model.predict(X_validation)
        probabilities = (
            model.predict_proba(X_validation)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        metrics = {
            "accuracy": float(accuracy_score(y_validation, predictions)),
            "precision": float(
                precision_score(y_validation, predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y_validation, predictions, zero_division=0)
            ),
            "f1_score": float(f1_score(y_validation, predictions, zero_division=0)),
        }
        if probabilities is not None:
            metrics["roc_auc"] = float(roc_auc_score(y_validation, probabilities))

        matrix = confusion_matrix(y_validation, predictions).tolist()
        return metrics, matrix

    @staticmethod
    def _feature_importance(model, feature_columns):
        if hasattr(model, "feature_importances_"):
            return {
                feature: float(importance)
                for feature, importance in zip(
                    feature_columns,
                    model.feature_importances_,
                )
            }
        if hasattr(model, "coef_"):
            return {
                feature: float(abs(coefficient))
                for feature, coefficient in zip(
                    feature_columns,
                    model.coef_[0],
                )
            }
        return {}

    def train_models(self, train_df, validation_df, experiment_name):
        mlflow.set_experiment(experiment_name)
        X_train, X_validation, y_train, y_validation = self.prepare_features(
            train_df,
            validation_df,
        )

        results = {}
        for model_name, model in self.models.items():
            with mlflow.start_run(run_name=model_name, nested=True):
                logging.info("Training %s", model_name)
                mlflow.log_params(self._safe_params(model))
                mlflow.log_param("model_type", model_name)
                mlflow.log_param("features_used", ",".join(self.FEATURE_COLUMNS))
                mlflow.log_metric("training_samples", len(X_train))
                mlflow.log_metric("validation_samples", len(X_validation))

                model.fit(X_train, y_train)
                metrics, matrix = self._evaluate_model(
                    model,
                    X_validation,
                    y_validation,
                )
                importance = self._feature_importance(model, self.FEATURE_COLUMNS)

                mlflow.log_metrics(metrics)
                mlflow.log_dict(
                    {
                        "labels": [0, 1],
                        "matrix": matrix,
                    },
                    f"confusion_matrix_{model_name}.json",
                )
                if importance:
                    mlflow.log_dict(
                        importance,
                        f"feature_importance_{model_name}.json",
                    )
                mlflow.sklearn.log_model(model, model_name)

                results[model_name] = {
                    "metrics": metrics,
                    "run_id": mlflow.active_run().info.run_id,
                    "model": model,
                }
                logging.info(
                    "%s: accuracy=%.4f, f1=%.4f",
                    model_name,
                    metrics["accuracy"],
                    metrics["f1_score"],
                )

        return results

    @staticmethod
    def find_best_model(results, metric="f1_score"):
        if not results:
            raise ValueError("Результаты обучения пусты.")
        best_model_name = max(
            results,
            key=lambda name: results[name]["metrics"].get(metric, float("-inf")),
        )
        return best_model_name, results[best_model_name]

    @staticmethod
    def save_comparison(results, best_model_name, path):
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison = {
            "best_model": best_model_name,
            "selection_metric": "f1_score",
            "models": {
                name: {
                    "metrics": result["metrics"],
                    "run_id": result["run_id"],
                }
                for name, result in results.items()
            },
        }
        output_path.write_text(
            json.dumps(comparison, indent=2),
            encoding="utf-8",
        )
        return comparison

