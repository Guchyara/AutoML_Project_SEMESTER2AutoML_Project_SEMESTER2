import logging

from sklearn.utils.multiclass import type_of_target


class TaskDetector:
    """Infer classification/regression from the target column."""

    def detect_task_type(self, target_series):
        detected = type_of_target(target_series)
        if detected in {"binary", "multiclass", "multiclass-multioutput"}:
            task_type = "classification"
        elif detected in {"continuous", "continuous-multioutput"}:
            task_type = "regression"
        else:
            # Integer targets with many distinct values are usually regression.
            task_type = "regression" if target_series.nunique() > 10 else "classification"
        logging.info("Target type=%s; detected task=%s", detected, task_type)
        return task_type

    def get_metrics(self, task_type):
        if task_type == "classification":
            return ["accuracy", "f1", "precision", "recall"]
        return ["rmse", "mae", "r2"]

