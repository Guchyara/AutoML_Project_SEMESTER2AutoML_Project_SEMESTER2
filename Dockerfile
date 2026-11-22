FROM apache/airflow:2.7.0-python3.10

COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir \
    "pandas==2.0.3" \
    "scikit-learn==1.3.0" \
    "numpy==1.24.3" \
    "kaggle==1.6.11" \
    "mlflow==2.8.1" \
    "python-dotenv==1.0.0"
