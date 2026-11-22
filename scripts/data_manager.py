import json
import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import pandas as pd


class DynamicDataManager:
    """Load and preprocess datasets selected at runtime by dataset name."""

    def __init__(self, config_path=None, data_root=None):
        self.config_path = Path(
            config_path or os.getenv("DYNAMIC_CONFIG_PATH", "/opt/airflow/config/dataset_configs.json")
        )
        self.data_root = Path(data_root or os.getenv("DYNAMIC_DATA_ROOT", "/opt/airflow/data"))
        self.raw_root = self.data_root / "dynamic_raw"
        self.processed_root = self.data_root / "dynamic_processed"
        self.supported_datasets = self._load_dataset_configs()

    def _load_dataset_configs(self):
        with self.config_path.open(encoding="utf-8") as config_file:
            return json.load(config_file)

    def get_config(self, dataset_name):
        if dataset_name not in self.supported_datasets:
            raise ValueError(
                f"Dataset {dataset_name!r} не поддерживается. "
                f"Доступны: {sorted(self.supported_datasets)}"
            )
        return self.supported_datasets[dataset_name]

    @staticmethod
    def _canonicalize_columns(dataframe, config):
        target = config["target_column"]
        aliases = {str(target).strip().lower(): target}
        if target.lower() == "species":
            aliases.update({"species": target, "variety": target, "class": target})
        renamed = {}
        for column in dataframe.columns:
            normalized = str(column).strip().lower().replace(" ", "_")
            renamed[column] = aliases.get(normalized, column)
        return dataframe.rename(columns=renamed)

    def _candidate_csv(self, directory, target_column):
        csv_files = sorted(Path(directory).rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"В {directory} не найден CSV-файл")
        for csv_path in csv_files:
            try:
                sample = pd.read_csv(csv_path, nrows=5)
                if target_column.lower() in {str(c).lower() for c in sample.columns}:
                    return csv_path
            except Exception as error:
                logging.warning("Не удалось прочитать %s: %s", csv_path, error)
        return csv_files[0]

    def _load_builtin(self, dataset_name):
        from sklearn.datasets import load_diabetes, load_iris

        fallback = self.get_config(dataset_name).get("builtin_fallback")
        if fallback == "iris":
            bunch = load_iris(as_frame=True)
            frame = bunch.frame.rename(columns={"target": "species"})
            return frame
        if fallback == "wine":
            # A deterministic local fallback with the same supervised shape.
            from sklearn.datasets import load_wine

            bunch = load_wine(as_frame=True)
            return bunch.frame.rename(columns={"target": "quality"})
        if fallback == "titanic":
            train_path = Path("/opt/airflow/data/raw/train.csv")
            if train_path.exists():
                return pd.read_csv(train_path)
        if fallback == "diabetes":
            bunch = load_diabetes(as_frame=True)
            return bunch.frame.rename(columns={"target": "price"})
        raise FileNotFoundError(f"Локальный fallback для {dataset_name!r} недоступен")

    def _load_kaggle(self, dataset_name, config, target_column, destination):
        from kaggle.api.kaggle_api_extended import KaggleApi

        credentials = Path("/home/airflow/.kaggle/kaggle.json")
        if not credentials.exists():
            raise FileNotFoundError("Не найден /home/airflow/.kaggle/kaggle.json")
        temporary_dir = Path(tempfile.mkdtemp(prefix="dynamic_kaggle_config_"))
        temporary_credentials = temporary_dir / "kaggle.json"
        shutil.copy2(credentials, temporary_credentials)
        temporary_credentials.chmod(0o600)
        os.environ["KAGGLE_CONFIG_DIR"] = str(temporary_dir)
        try:
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(
                config["kaggle_dataset"], path=str(destination), unzip=True
            )
            candidate = self._candidate_csv(destination, target_column)
        except Exception:
            competition = config.get("kaggle_competition")
            if not competition:
                raise
            api.competition_download_files(
                competition, path=str(destination), quiet=False, force=True
            )
            archive_path = destination / f"{competition}.zip"
            if archive_path.exists():
                with zipfile.ZipFile(archive_path) as archive:
                    archive.extractall(destination)
            candidate = self._candidate_csv(destination, target_column)
        finally:
            shutil.rmtree(temporary_dir, ignore_errors=True)
        return pd.read_csv(candidate)

    def download_dataset(self, dataset_name):
        config = self.get_config(dataset_name)
        target_column = config["target_column"]
        destination = self.raw_root / dataset_name
        destination.mkdir(parents=True, exist_ok=True)
        output_path = destination / "raw.csv"
        if output_path.exists():
            logging.info("Используется кэш динамического датасета: %s", output_path)
            return str(output_path)

        source_mode = os.getenv("DYNAMIC_DATA_SOURCE", "auto").strip().lower()
        if source_mode not in {"auto", "local", "kaggle"}:
            raise ValueError("DYNAMIC_DATA_SOURCE должен быть auto, local или kaggle")
        allow_builtin = os.getenv("DYNAMIC_ALLOW_BUILTIN_FALLBACK", "true").lower() == "true"
        try:
            if source_mode == "local" and dataset_name == "titanic" and Path("/opt/airflow/data/raw/train.csv").exists():
                dataframe = pd.read_csv("/opt/airflow/data/raw/train.csv")
                source = "local_kaggle_cache"
            elif source_mode == "local":
                dataframe = self._load_builtin(dataset_name)
                source = "sklearn_builtin"
            elif source_mode == "auto" and dataset_name == "titanic" and Path("/opt/airflow/data/raw/train.csv").exists():
                dataframe = pd.read_csv("/opt/airflow/data/raw/train.csv")
                source = "local_kaggle_cache"
            else:
                dataframe = self._load_kaggle(dataset_name, config, target_column, destination)
                source = "kaggle"
        except Exception as error:
            if not allow_builtin or not config.get("builtin_fallback"):
                raise
            logging.warning("Kaggle не загрузил %s (%s); используется локальный fallback", dataset_name, error)
            dataframe = self._load_builtin(dataset_name)
            source = "sklearn_builtin"

        dataframe = self._canonicalize_columns(dataframe, config)
        self.validate_dataset(dataframe, dataset_name)
        dataframe.to_csv(output_path, index=False)
        (destination / "metadata.json").write_text(
            json.dumps({"dataset": dataset_name, "source": source, "rows": len(dataframe)}, indent=2),
            encoding="utf-8",
        )
        logging.info("Датасет %s загружен из %s: %s", dataset_name, source, dataframe.shape)
        return str(output_path)

    def preprocess_data(self, dataframe, dataset_name):
        config = self.get_config(dataset_name)
        target = config["target_column"]
        dataframe = self._canonicalize_columns(dataframe.copy(), config)
        if target not in dataframe.columns:
            raise ValueError(f"В датасете {dataset_name} отсутствует target {target}")
        dataframe = dataframe.dropna(subset=[target]).reset_index(drop=True)

        for column in list(dataframe.columns):
            if column == target:
                continue
            if dataframe[column].nunique(dropna=False) == len(dataframe):
                dataframe = dataframe.drop(columns=[column])

        for column in dataframe.columns:
            if column == target:
                continue
            if pd.api.types.is_numeric_dtype(dataframe[column]):
                dataframe[column] = dataframe[column].fillna(dataframe[column].median())
            else:
                mode = dataframe[column].mode(dropna=True)
                dataframe[column] = dataframe[column].fillna(mode.iloc[0] if not mode.empty else "unknown")
                dataframe[column] = pd.factorize(dataframe[column].astype(str))[0]

        if config["problem_type"] == "regression":
            dataframe[target] = pd.to_numeric(dataframe[target], errors="raise")
        else:
            if not pd.api.types.is_numeric_dtype(dataframe[target]):
                dataframe[target] = pd.factorize(dataframe[target].astype(str))[0]
        return dataframe

    def validate_dataset(self, dataframe, dataset_name):
        config = self.get_config(dataset_name)
        if config["target_column"] not in dataframe.columns:
            raise ValueError(
                f"Целевая колонка {config['target_column']!r} не найдена в {dataset_name}: "
                f"{list(dataframe.columns)}"
            )
        if len(dataframe) < 10:
            raise ValueError(f"Датасет {dataset_name} слишком мал: {len(dataframe)} строк")
        return True

