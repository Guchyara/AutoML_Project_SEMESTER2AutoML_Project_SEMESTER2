# Контекст лабораторной работы MLOps: Titanic, Airflow и MLflow

## 1. Цель работы

Реализовать базовый MLOps-пайплайн для датасета Titanic с использованием
Apache Airflow и MLflow.

Пайплайн должен: 1. Скачать датасет Titanic с Kaggle. 2. Выполнить
предварительную обработку данных. 3. Сформировать сводную информацию о
датасете и записать результаты/метрики в MLflow.

## 2. Среда разработки

Работа выполняется на Windows 11. В качестве среды запуска используется
Docker Desktop с Docker Compose, а для редактирования проекта --- Pycharm Python

Используемые компоненты и версии, обсуждавшиеся в чате: - Apache Airflow
2.7.0 (образ `apache/airflow:2.7.0-python3.10`); - Python 3.10; - MLflow
2.8.1; - pandas; - scikit-learn; - Kaggle API; - PostgreSQL для
метаданных Airflow.

## 3. Предполагаемая структура проекта

``` text
lab1_titanic/
├── dags/
│   └── titanic_basic_pipeline.py
├── scripts/
│   ├── data_loader.py
│   └── data_preprocessor.py
├── data/
│   └── raw/
├── kaggle/
│   └── kaggle.json
├── mlflow_data/
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

Папка с проектом лежит в  c:\Users\mrngu\AutoML_PROJ

Назначение основных файлов: - `dags/titanic_basic_pipeline.py` ---
описание DAG и задач Airflow. - `scripts/data_loader.py` --- загрузка
Titanic с Kaggle и чтение CSV. - `scripts/data_preprocessor.py` ---
предварительная обработка данных. - `Dockerfile` --- сборка образа
Airflow с необходимыми Python-библиотеками. - `docker-compose.yml` ---
запуск Airflow, PostgreSQL и MLflow. - `kaggle/kaggle.json` ---
API-токен Kaggle. Это секретный файл, его нельзя публиковать или
добавлять в открытый репозиторий.

## 4. Архитектура Docker Compose

В обсуждавшейся конфигурации предусмотрены следующие сервисы: -
`postgres` --- база метаданных Airflow; - `airflow-init` ---
первоначальная инициализация базы Airflow и создание пользователя; -
`airflow-webserver` --- веб-интерфейс Airflow; - `airflow-scheduler` ---
планировщик и запуск задач; - `mlflow` --- сервер отслеживания
экспериментов.

Для Airflow обсуждались следующие настройки: - исполнитель
`LocalExecutor`; - подключение к PostgreSQL через
`AIRFLOW__DATABASE__SQL_ALCHEMY_CONN`; - отключение примеров DAG; -
каталог DAG `/opt/airflow/dags`; - автоматический запуск DAG без
состояния паузы через
`AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=False`.

Важные подключения томов (volumes), которые должны быть согласованы с
фактическим `docker-compose.yml`: - `./dags:/opt/airflow/dags`; -
`./scripts:/opt/airflow/scripts`; - `./data:/opt/airflow/data`; -
`./kaggle:/home/airflow/.kaggle:ro`.

MLflow в обсуждавшемся варианте запускается на порту 5000, использует
SQLite backend store и каталог артефактов внутри подключённого тома
`mlflow_data`.

> Примечание: это описание отражает обсуждавшуюся конфигурацию. Перед
> запуском следует сверить фактические имена сервисов, переменные
> окружения и volumes в локальном `docker-compose.yml`.

## 5. DAG и последовательность задач

DAG называется `titanic_basic_pipeline`. В нём предусмотрены три задачи:

1.  `download_titanic_data` --- скачивание датасета с Kaggle и чтение
    исходных таблиц.
2.  `preprocess_titanic_data` --- очистка и преобразование данных.
3.  `log_dataset_summary` --- логирование сводной информации о данных в
    MLflow.

Ожидаемая последовательность:

``` text
download_titanic_data
        |
        v
preprocess_titanic_data
        |
        v
log_dataset_summary
```

Для связи DAG со скриптами используется каталог `/opt/airflow/scripts`.
Если Airflow не находит модуль `data_loader`, в начале DAG-файла перед
импортами можно добавить:

``` python
import sys

SCRIPTS_DIR = "/opt/airflow/scripts"
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
```

И проверить, что каталог `./scripts` действительно подключён в контейнер
как `/opt/airflow/scripts`.

## 6. Загрузка данных с Kaggle

В чате использовался датасет: - Kaggle dataset: `heptapod/titanic` -
Страница: https://www.kaggle.com/datasets/heptapod/titanic

Для API требуется файл `kaggle.json`, получаемый в настройках аккаунта
Kaggle в разделе API. В Docker Compose папка с этим файлом подключается
в контейнер Airflow, например:

``` yaml
- ./kaggle:/home/airflow/.kaggle:ro
```

Ожидаемая папка данных внутри контейнера: `/opt/airflow/data/raw`

Ожидаемые CSV: - `/opt/airflow/data/raw/train.csv` -
`/opt/airflow/data/raw/test.csv`

Вручную помещать CSV в `data` не предполагалось: по замыслу их должен
скачивать первый шаг DAG. Если автоматическое скачивание не работает,
ручное помещение файлов может использоваться только как временная
диагностика/обходной путь, но это не заменяет исправление загрузчика.

## 7. Предварительная обработка

В обсуждавшемся варианте предварительная обработка включает: -
заполнение пропусков в `Age` медианным значением; - заполнение пропусков
в `Fare` медианным значением; - заполнение пропусков в `Embarked`
наиболее частым значением; - создание признака `FamilySize`; - создание
признака `IsAlone`; - кодирование категориальных признаков `Sex` и
`Embarked` с помощью `LabelEncoder`; - разделение/подготовку обучающей и
тестовой выборок.

Точный перечень признаков, правила обработки и формат выходных файлов
нужно сверить с текущим содержимым `scripts/data_preprocessor.py`.

## 8. Логирование в MLflow

DAG настраивает tracking URI на адрес сервиса внутри Docker-сети:

``` python
mlflow.set_tracking_uri("http://mlflow:5000")
```

Задачи должны записывать в MLflow сведения о датасете и/или результатах
обработки. В первоначальном варианте обсуждалось логирование сводных
данных и метрик в отдельных MLflow runs.

Адрес `http://mlflow:5000` предназначен для взаимодействия контейнеров
внутри Docker Compose. Для доступа из браузера Windows обычно
используется опубликованный порт сервиса, если он задан в Compose.

## 9. Что уже возникало при настройке

### 9.1. Не включена Virtual Machine Platform / WSL

Docker Desktop сообщал, что не включена необходимая виртуализация
Windows. Для решения обсуждались установка/обновление WSL, включение
необходимых компонентов Windows и перезагрузка.

Полезные команды PowerShell:

``` powershell
wsl --status
wsl --list --verbose
wsl --update
wsl --shutdown
```

### 9.2. Docker Engine недоступен

Появлялась ошибка подключения к Docker API через Windows named pipe
(`docker_engine`). Это указывало на то, что Docker Desktop/Engine не
запущен или ещё не завершил запуск.

Проверки:

``` powershell
docker info
docker context ls
```

Нужно дождаться состояния Docker Desktop, в котором Engine запущен.

### 9.3. Broken DAG: не найден `data_loader`

Ошибка:

``` text
ModuleNotFoundError: No module named 'data_loader'
```

Обсуждавшееся решение: 1. Добавить `/opt/airflow/scripts` в `sys.path` в
начале DAG-файла. 2. Убедиться, что в Compose есть volume
`./scripts:/opt/airflow/scripts`. 3. Проверить содержимое каталога в
контейнере:

``` powershell
docker compose exec airflow-scheduler ls -la /opt/airflow/scripts
```

4.  Перезапустить сервисы:

``` powershell
docker compose restart airflow-scheduler airflow-webserver
```

### 9.4. Текущая ошибка загрузки Titanic

Последний предоставленный лог Airflow относится к задаче
`download_titanic_data`.

Ключевые строки:

``` text
Dataset URL: https://www.kaggle.com/datasets/heptapod/titanic
...
FileNotFoundError: [Errno 2] No such file or directory:
'/opt/airflow/data/raw/train.csv'
```

Стек указывает на: - DAG: `/opt/airflow/dags/titanic_basic_pipeline.py`,
функция `download_and_log_data`; - загрузчик:
`/opt/airflow/scripts/data_loader.py`, строка с `pd.read_csv(...)`.

Это означает, что выполнение дошло до чтения CSV, но файл по ожидаемому
пути отсутствует. Сам лог не доказывает, что Kaggle API завершил
скачивание и распаковку в ожидаемую папку. Также в логе есть
предупреждение о правах доступа к `kaggle.json`, но оно не является
показанной причиной падения.

В ответ на эту ошибку было предложено: - создать папку назначения через
`os.makedirs(DATA_DIR, exist_ok=True)`; - скачать датасет с
`api.dataset_download_files("heptapod/titanic", path=DATA_DIR, unzip=True)`; -
вывести фактическую структуру каталогов через `os.walk(DATA_DIR)`; -
проверить наличие `train.csv` и `test.csv` до вызова `pd.read_csv`.

Пример предложенного диагностического загрузчика:

``` python
import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

DATA_DIR = "/opt/airflow/data/raw"

def download_titanic_data():
    os.makedirs(DATA_DIR, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(
        "heptapod/titanic",
        path=DATA_DIR,
        unzip=True
    )

    for root, dirs, files in os.walk(DATA_DIR):
        print(f"Папка: {root}; файлы: {files}")

    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Не найден {train_path}. Проверь структуру скачанного датасета."
        )

    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Не найден {test_path}. Проверь структуру скачанного датасета."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df
```

Это предложенный вариант исправления, но его результат ещё не
подтверждён пользователем. Если CSV окажутся во вложенной папке или
Kaggle вернёт иной набор файлов, путь следует адаптировать по выводу
`os.walk`.

После изменения кода предлагалось перезапустить:

``` powershell
docker compose restart airflow-scheduler airflow-webserver
```

Затем повторно запустить DAG и при необходимости прислать новые логи.

## 10. Текущий статус и ближайшие действия

На момент создания этого документа: - Docker Desktop и Airflow удалось
запустить; - DAG обнаруживается Airflow; - проблему импорта
`data_loader` обсуждали и предложили исправить через `sys.path` и
volume; - текущий подтверждённый сбой --- отсутствие
`/opt/airflow/data/raw/train.csv` при выполнении задачи
`download_titanic_data`; - предложенный диагностический вариант
загрузчика ещё не подтверждён успешным запуском.

Рекомендуемый следующий порядок: 1. Проверить фактический
`scripts/data_loader.py`. 2. Убедиться, что `./data:/opt/airflow/data`
подключён к scheduler и webserver/worker-контейнерам, выполняющим
задачи. 3. Проверить, что `kaggle.json` доступен внутри контейнера и
API-аутентификация проходит. 4. Запустить задачу загрузки с
диагностическим выводом файлов после скачивания и распаковки. 5. По
фактической структуре файлов скорректировать пути к CSV. 6. После
успешной загрузки проверить выполнение `preprocess_titanic_data`. 7.
Затем проверить появление эксперимента и run в MLflow.

## 11. Полезные команды Docker Compose

Выполнять из PowerShell в каталоге проекта:

``` powershell
# Запустить сервисы
docker compose up -d

# Посмотреть состояние сервисов
docker compose ps

# Посмотреть логи scheduler
docker compose logs --tail=100 airflow-scheduler

# Проверить файлы скриптов внутри контейнера
docker compose exec airflow-scheduler ls -la /opt/airflow/scripts

# Проверить каталог данных внутри контейнера
docker compose exec airflow-scheduler ls -la /opt/airflow/data/raw

# Перезапустить Airflow
docker compose restart airflow-scheduler airflow-webserver
```

Названия сервисов в командах должны совпадать с фактическим
`docker-compose.yml`.

------------------------------------------------------------------------

**Важно о безопасности:** не отправляйте в чат содержимое `kaggle.json`,
пароли, токены и секретные ключи. Если токен случайно опубликован,
отзовите его в настройках Kaggle и создайте новый.

## 12. Проверка проекта на 2026-11-15

Проверены файлы в `C:\Users\mrngu\AutoML_PROJ`.

- Фактические volumes в `docker-compose.yml` подключают `dags`, `scripts`, `data` и `kaggle` ко всем контейнерам Airflow.
- `dags/titanic_basic_pipeline.py` добавляет `/opt/airflow/scripts` в `sys.path`, поэтому прежняя проблема импорта `data_loader` устранена в исходном коде.
- В локальном `data/raw` находится файл `train_and_test2.csv` размером 1309 строк × 28 колонок. Его схема отличается от ожидаемой схемы Titanic (`train.csv`/`test.csv`): используются нестандартные имена (`Passengerid`, `sibsp`, `2urvived`, `zero...`) и уже числовое кодирование. Файл не считается подтверждённым результатом текущего Kaggle-загрузчика.
- Текущий `scripts/data_loader.py` без дополнительной диагностики сразу открывает `/opt/airflow/data/raw/train.csv` и `/opt/airflow/data/raw/test.csv`. При отсутствии этих файлов ошибка возникает как `FileNotFoundError` и не показывает фактическое содержимое каталога после скачивания.
- На момент проверки Docker CLI был установлен, но Docker Engine недоступен через `dockerDesktopLinuxEngine`; запуск DAG в контейнерах до восстановления Docker Desktop невозможен.

### Решения, ожидающие подтверждения

1. Основным источником данных считать Kaggle `heptapod/titanic`, а локальный `train_and_test2.csv` оставить диагностическим артефактом.
2. Сохранить объём лабораторной из разделов 1 и 5: загрузка, предобработка и логирование сводки в MLflow.
3. Вести дальнейшие обновления этого рабочего контекста в проекте, не изменяя исходный файл в `Downloads`.

## 13. Принятые решения и изменения

- Источник данных поддерживает три явных режима через `TITANIC_DATA_SOURCE`: `kaggle`, `local` и `auto`; по умолчанию используется `kaggle`.
- Режим `local` принимает пару `train.csv`/`test.csv` в `data/raw` либо объединённый файл из `TITANIC_LOCAL_DATA_PATH`. Для текущего `train_and_test2.csv` размер разделения задаётся `TITANIC_LOCAL_TRAIN_ROWS` и по умолчанию равен 891.
- Загрузчик теперь ищет CSV рекурсивно, выводит список фактически найденных файлов после Kaggle-скачивания, проверяет структуру и копирует read-only `kaggle.json` во временный каталог с правами 600.
- Предобработка проверяет обязательные поля, приводит числовые признаки к числовому типу и выдаёт понятные ошибки для неподдерживаемой схемы.
- MLflow настраивается во время выполнения задач, а не при импорте DAG. В Compose добавлен healthcheck MLflow и ожидание его готовности для scheduler/webserver.
- Объём проекта оставлен прежним: загрузка, предобработка и логирование сводной информации в MLflow.

- Добавлен `.env.example` с примерами переключения режима данных и параметрами локального объединённого файла.

### Переключение источника данных

Перед запуском Compose можно выбрать режим в `.env` (или переменными окружения):

```env
TITANIC_DATA_SOURCE=kaggle
```

Для локального файла:

```env
TITANIC_DATA_SOURCE=local
TITANIC_LOCAL_DATA_PATH=/opt/airflow/data/raw/train_and_test2.csv
TITANIC_LOCAL_TRAIN_ROWS=891
```

Режим `auto` сначала ищет локальный источник, затем использует Kaggle:

```env
TITANIC_DATA_SOURCE=auto
```

## 14. Результат локальной проверки

15.11.2026 выполнен прогон в Docker с `TITANIC_DATA_SOURCE=local` на текущем `train_and_test2.csv`:

- `download_titanic_data` — success; получены train 891×9 и test 418×8.
- `preprocess_titanic_data` — success; записаны `data/final/train.csv` и `data/final/test.csv`.
- `log_dataset_summary` — success.
- В MLflow создан эксперимент `titanic_basic_pipeline_v2` с artifact location `mlflow-artifacts:/2` и завершены три run: `data_download`, `data_preprocessing`, `dataset_summary`.
- В run `data_preprocessing` присутствует артефакт `processed_data_sample.csv`.
- MLflow запускается с `--serve-artifacts` и `--artifacts-destination /mlflow/artifacts`, чтобы Airflow отправлял артефакты через tracking server.
- После проверки `.env` очищен; при следующем запуске Compose режим по умолчанию снова `kaggle`. Для локального прогона используйте параметры из `.env.example`.

Первоначальные неуспешные validation-runs оставлены в истории Airflow как диагностические записи; успешный запуск — `local-validation-3`.

## 15. Проверка Kaggle-режима

- API для `heptapod/titanic` вернул пустой список файлов. Поиск подтвердил, что стандартные `train.csv` и `test.csv` доступны в Kaggle competition `titanic`.
- Загрузчик теперь сначала пробует `KAGGLE_DATASET`, затем `KAGGLE_COMPETITION`; competition ZIP распаковывается явно.
- Прогон `kaggle-validation-3` в Docker завершён успешно по всем трём задачам.
- В `data/raw` после этого появились архив `titanic.zip`, `train.csv` и `test.csv`; старый `train_and_test2.csv` сохранён для локального режима.

## 16. Задание 2: сравнительный анализ моделей с MLflow

Для задания 2 добавлены:

- `scripts/model_trainer.py` — обучение и сравнение Logistic Regression, Random Forest и SVM;
- `dags/titanic_model_training.py` — полный DAG из шести задач: загрузка, предобработка, stratified train/validation split, обучение моделей, регистрация лучшей модели и отчёт.

Каждая модель логируется во вложенный MLflow run. Для моделей сохраняются параметры, accuracy, precision, recall, F1, ROC-AUC, матрица ошибок, доступная важность признаков и артефакт модели. Сравнение записывается в `data/model_results/model_comparison.json`.

15.11.2026 выполнен запуск `model-comparison-validation`:

- все шесть задач DAG завершились со статусом `success`;
- по F1 лидирует `random_forest` (`0.7519`), далее Logistic Regression (`0.7344`) и SVM (`0.3232`);
- лучшая модель зарегистрирована как `titanic-survival-predictor`, версия `1` имеет статус `READY` и стадию `Production`;
- MLflow использует эксперимент `titanic_model_comparison` с artifact location `mlflow-artifacts:/3`.

Для воспроизводимости добавлена фиксированная зависимость `numpy==1.24.3`; образ Airflow пересобран перед валидационным запуском.

Ключевые скриншоты этого этапа сохранены в папке `C:\Users\mrngu\OneDrive\Namizje\Учёба\2026-2027\Автоматизация машинного обучения\Скриншоты` с продолжением нумерации: `Скрин_13`–`Скрин_16`.

## 17. Задание 3: динамический пайплайн для нескольких датасетов

Добавлены:

- `config/dataset_configs.json` с конфигурациями `titanic`, `iris`, `wine_quality` и `house_prices`;
- `scripts/data_manager.py` для загрузки из Kaggle, локального кэша и безопасного встроенного fallback;
- `scripts/task_detector.py` для определения классификации или регрессии по целевой переменной;
- `scripts/automl_trainer.py` для выбора моделей по типу задачи, метрик и логирования в MLflow;
- `scripts/mlflow_client_hook.py` с `MLflowClientHook` для поиска экспериментов и регистрации моделей;
- `dags/dynamic_ml_pipeline.py` с TaskFlow API (`@task`) и dynamic task mapping.

Список датасетов задаётся Airflow Variable `dynamic_datasets` в формате JSON, например:

```json
["titanic", "iris"]
```

15.11.2026 выполнен запуск `dynamic-validation-4` для `titanic` и `iris`:

- все mapped-задачи загрузки, предобработки, обучения и регистрации завершились успешно;
- для каждого датасета были созданы отдельные MLflow experiment и зарегистрированная модель;
- для Titanic лучшей стала `gradient_boosting` с F1 `0.8068`;
- для Iris лучшей стала `logistic_regression` с accuracy `0.9667`;
- модели `dynamic-titanic-predictor` и `dynamic-iris-predictor` имеют версии `2` в стадии `Production`;
- общий отчёт записан в `data/dynamic_results/dynamic_pipeline_report.json`.

Для текущего этапа скриншоты выполняются пользователем вручную; ключевые места указаны в ответе после проверки.

## 18. Задание 4: end-to-end MLOps с развёртыванием

Добавлены:

- `api/fastapi_app.py` — FastAPI-сервис с `/health`, `/model-info`, `/predict`, `/reload-model` и `/metrics`;
- `Dockerfile.api` — отдельный образ inference API;
- `scripts/model_load_predict_operator.py` — Airflow-оператор `ModelLoadAndPredictOperator`, загружающий модель из MLflow Registry;
- `dags/mlops_inference_pipeline.py` — подготовка batch, проверка API, inference через API и прямой inference через MLflow Operator;
- `scripts/data_drift_detector.py` и `scripts/model_retraining.py` — лёгкий мониторинг дрейфа и условное переобучение;
- `dags/mlops_monitoring_pipeline.py` — мониторинг и решение о переобучении;
- `prometheus.yml`, Prometheus и Grafana в `docker-compose.yml`.

Проверки 15.11.2026:

- FastAPI контейнер healthy; Production-модель `titanic-survival-predictor` загружается через `models:/titanic-survival-predictor/Production`;
- запуск `inference-validation-2` завершён успешно по всем четырём задачам;
- batch содержит 418 строк, результаты API и `ModelLoadAndPredictOperator` совпали полностью;
- результаты сохранены в `data/inference/api_results.csv`, `data/inference/operator_results.csv` и SQLite-хранилище `data/inference/inference_results.db` (`predictions` и `operator_predictions`, по 418 строк);
- Prometheus endpoint API отдаёт счётчики запросов и предсказаний;
- запуск `monitoring-validation` завершён успешно;
- дрейф не обнаружен: `drift_score=0.1629`, доля изменившихся признаков `0.0`, переобучение пропущено по условию.

Тяжёлые Evidently/Alibi зависимости не добавлялись: мониторинг реализован на стандартных pandas и MLflow, чтобы сохранить совместимость текущего Docker-образа.

На момент завершения предыдущего этапа в пользовательской папке скриншотов присутствуют файлы `Скрин_17`–`Скрин_21`; для задания 4 рекомендуется продолжить нумерацию с `Скрин_22`.
