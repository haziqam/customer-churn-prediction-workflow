from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime

with DAG(
    dag_id='experiment_dag_minio',
    schedule_interval=None,
    start_date=datetime(2025, 1, 8),
    catchup=False,
) as dag:

    ingest_training_data_task = SparkSubmitOperator(
        task_id='ingest_training_data',
        application='/opt/airflow/functions/ingest_training_data.py',
        conn_id='spark_default',
        name='ingest_training_data_job',
        application_args=[
            '/opt/workspace/dataset.csv',  # Input path
            'my-bucket',  # MinIO bucket name
            'ingested_data.csv',  # MinIO object key
        ],
        conf={
            'spark.executor.memory': '512m',
        },
        verbose=True,
    )

    preprocess_task = SparkSubmitOperator(
        task_id='preprocess',
        application='/opt/airflow/functions/preprocess.py',
        conn_id='spark_default',
        name='preprocess_job',
        application_args=[
            'my-bucket',  # MinIO bucket name
            'ingested_data.csv',  # Input object key
            'preprocessed_data.csv',  # Output object key
        ],
        conf={
            'spark.executor.memory': '512m',
        },
        verbose=True,
    )

    ingest_training_data_task >> preprocess_task
