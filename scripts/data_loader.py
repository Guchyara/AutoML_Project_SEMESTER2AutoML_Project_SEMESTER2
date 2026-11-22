import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import pandas as pd


DATA_DIR = Path(os.getenv("TITANIC_DATA_DIR", "/opt/airflow/data"))
RAW_DIR = DATA_DIR / "raw"
DEFAULT_LOCAL_FILE = RAW_DIR / "train_and_test2.csv"
DATA_SOURCE = os.getenv("TITANIC_DATA_SOURCE", "kaggle").strip().lower()
KAGGLE_DATASET = os.getenv("KAGGLE_DATASET", "heptapod/titanic")
KAGGLE_COMPETITION = os.getenv("KAGGLE_COMPETITION", "titanic")


def _canonicalize_columns(dataframe):
    """Convert known aliases to the schema expected by preprocessing."""
    aliases = {
        "passengerid": "PassengerId",
        "passenger_id": "PassengerId",
        "age": "Age",
        "fare": "Fare",
        "sex": "Sex",
        "sibsp": "SibSp",
        "sibling_spouse": "SibSp",
        "parch": "Parch",
        "parent_child": "Parch",
        "pclass": "Pclass",
        "embarked": "Embarked",
        "survived": "Survived",
        "2urvived": "Survived",
    }

    renamed = {}
    for column in dataframe.columns:
        normalized = str(column).strip().lower()
        renamed[column] = aliases.get(normalized, column)

    result = dataframe.rename(columns=renamed)
    zero_columns = [
        column
        for column in result.columns
        if str(column).strip().lower().startswith("zero")
    ]
    if zero_columns:
        result = result.drop(columns=zero_columns)
        logging.info("Удалены служебные колонки локального файла: %s", zero_columns)

    return result


def _find_split_files(directory):
    train_files = sorted(directory.rglob("train.csv"))
    test_files = sorted(directory.rglob("test.csv"))

    if not train_files or not test_files:
        return None

    if len(train_files) > 1 or len(test_files) > 1:
        logging.warning(
            "Найдено несколько вариантов train/test; используется первый: %s, %s",
            train_files[0],
            test_files[0],
        )

    return train_files[0], test_files[0]


def _load_local_data():
    """Load local split files or an explicitly configured combined file."""
    split_files = _find_split_files(RAW_DIR)
    if split_files:
        train_path, test_path = split_files
        logging.info("Локальный источник: train=%s, test=%s", train_path, test_path)
        return (
            _canonicalize_columns(pd.read_csv(train_path)),
            _canonicalize_columns(pd.read_csv(test_path)),
        )

    combined_path = Path(
        os.getenv("TITANIC_LOCAL_DATA_PATH", str(DEFAULT_LOCAL_FILE))
    )
    if not combined_path.exists():
        raise FileNotFoundError(
            "Локальный источник не найден. Ожидались train.csv/test.csv в "
            f"{RAW_DIR} или файл {combined_path}."
        )

    combined = _canonicalize_columns(pd.read_csv(combined_path))
    if "Survived" not in combined.columns:
        raise ValueError(
            f"В локальном файле {combined_path} отсутствует колонка Survived "
            "(поддерживаются также исходные имена survived и 2urvived)."
        )

    train_rows = int(os.getenv("TITANIC_LOCAL_TRAIN_ROWS", "891"))
    if not 0 < train_rows < len(combined):
        raise ValueError(
            "TITANIC_LOCAL_TRAIN_ROWS должен быть между 1 и числом строк "
            f"файла ({len(combined) - 1}). Получено: {train_rows}."
        )

    logging.warning(
        "Локальный объединённый файл разделяется по первым %s строкам; "
        "для другого формата задайте TITANIC_LOCAL_TRAIN_ROWS.",
        train_rows,
    )
    train_df = combined.iloc[:train_rows].copy()
    test_df = combined.iloc[train_rows:].copy()
    test_df = test_df.drop(columns=["Survived"])
    return train_df, test_df


def _copy_kaggle_config():
    """Copy read-only mounted credentials to a private writable directory."""
    configured_dir = os.getenv("KAGGLE_CONFIG_DIR")
    candidates = []
    if configured_dir:
        candidates.append(Path(configured_dir) / "kaggle.json")
    candidates.extend(
        [
            Path("/home/airflow/.kaggle/kaggle.json"),
            Path.home() / ".kaggle" / "kaggle.json",
        ]
    )

    source = next((candidate for candidate in candidates if candidate.exists()), None)
    if source is None:
        raise FileNotFoundError(
            "Не найден kaggle.json. Подключите ./kaggle к "
            "/home/airflow/.kaggle или задайте KAGGLE_CONFIG_DIR."
        )

    temporary_dir = Path(tempfile.mkdtemp(prefix="kaggle_config_"))
    target = temporary_dir / "kaggle.json"
    shutil.copy2(source, target)
    try:
        target.chmod(0o600)
    except OSError:
        logging.warning("Не удалось установить права 600 для временного kaggle.json")
    os.environ["KAGGLE_CONFIG_DIR"] = str(temporary_dir)
    return temporary_dir


def _log_raw_tree():
    files = sorted(path for path in RAW_DIR.rglob("*") if path.is_file())
    logging.info("Файлы в %s: %s", RAW_DIR, [str(path) for path in files])


def _load_kaggle_data():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    temporary_config = _copy_kaggle_config()
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        logging.info("Пробуем Kaggle dataset: %s", KAGGLE_DATASET)
        api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(RAW_DIR),
            unzip=True,
        )
        if not _find_split_files(RAW_DIR):
            logging.warning(
                "Dataset %s не содержит доступных train.csv/test.csv; "
                "пробуем Kaggle competition: %s",
                KAGGLE_DATASET,
                KAGGLE_COMPETITION,
            )
            api.competition_download_files(
                KAGGLE_COMPETITION,
                path=str(RAW_DIR),
                quiet=False,
                force=True,
            )
            archive_path = RAW_DIR / f"{KAGGLE_COMPETITION}.zip"
            if archive_path.exists():
                with zipfile.ZipFile(archive_path) as archive:
                    archive.extractall(RAW_DIR)
                logging.info("Распакован Kaggle archive: %s", archive_path)
    finally:
        shutil.rmtree(temporary_config, ignore_errors=True)

    _log_raw_tree()
    split_files = _find_split_files(RAW_DIR)
    if not split_files:
        raise FileNotFoundError(
            "Kaggle API завершил скачивание, но train.csv/test.csv не найдены "
            f"в {RAW_DIR}. Проверьте список файлов выше."
        )

    train_path, test_path = split_files
    return (
        _canonicalize_columns(pd.read_csv(train_path)),
        _canonicalize_columns(pd.read_csv(test_path)),
    )


def download_titanic_data():
    """Load Titanic data using the selected explicit source mode.

    TITANIC_DATA_SOURCE accepts kaggle, local or auto.
    auto uses local data when a usable local source exists and otherwise
    falls back to Kaggle.
    """
    if DATA_SOURCE not in {"kaggle", "local", "auto"}:
        raise ValueError(
            "TITANIC_DATA_SOURCE должен быть одним из: kaggle, local, auto. "
            f"Получено: {DATA_SOURCE}"
        )

    if DATA_SOURCE == "local":
        train_df, test_df = _load_local_data()
        effective_source = "local"
    elif DATA_SOURCE == "auto":
        try:
            train_df, test_df = _load_local_data()
            effective_source = "local"
        except (FileNotFoundError, ValueError) as error:
            logging.info("Локальный источник недоступен (%s); используется Kaggle", error)
            train_df, test_df = _load_kaggle_data()
            effective_source = "kaggle"
    else:
        train_df, test_df = _load_kaggle_data()
        effective_source = "kaggle"

    logging.info(
        "Загружены данные из %s: train=%s, test=%s",
        effective_source,
        train_df.shape,
        test_df.shape,
    )
    return train_df, test_df


def save_data_locally(train_df, test_df, path=None):
    output_dir = Path(path or DATA_DIR / "processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)


def load_data_from_local(path=None):
    input_dir = Path(path or DATA_DIR / "processed")
    train_path = input_dir / "train.csv"
    test_path = input_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Ожидались {train_path} и {test_path}. "
            "Сначала выполните задачу загрузки данных."
        )
    return pd.read_csv(train_path), pd.read_csv(test_path)

