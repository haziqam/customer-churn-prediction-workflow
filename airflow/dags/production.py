from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from functions.generate_filename import generate_filename
from functions.get_production_data import get_and_preprocess_production_data
from functions.detect_drift import detect_drift
from functions.ingest_clean_data import ingest_clean_data
from functions.train_model import train_model
from functions.branch_based_on_drift import branch_based_on_drift
from functions.get_last_used_dataset import get_last_used_dataest

DATASET_BUCKET_NAME = 'datasets'

with DAG(
    dag_id='production_dag',
    schedule_interval=None,
    start_date=datetime(2025, 1, 10),
    catchup=False
) as dag:
    generate_filename_task = PythonOperator(
        task_id='generate_production_filename',
        python_callable=generate_filename,
    )

    get_and_preprocess_production_data_task = SparkSubmitOperator(
        task_id='get_and_preprocess_production_data',
        application='/opt/airflow/dags/functions/get_production_data.py',
        conn_id='spark_default',
        name='preprocess_production_job',
        application_args=[
            DATASET_BUCKET_NAME,
            "{{ task_instance.xcom_pull(task_ids='generate_production_filename', key='filename') }}"
        ],
        conf={
            'spark.executor.memory': '512m',
        },
        verbose=True,
    )

    get_last_used_dataset_task = PythonOperator(
        task_id='get_last_used_dataset',
        python_callable=get_last_used_dataest,
    )

    detect_drift_task = PythonOperator(
        task_id='detect_drift',
        python_callable=detect_drift,
        op_args=[
            "{{ task_instance.xcom_pull(task_ids='get_last_used_dataset', key='last_used_dataset') }}",
            DATASET_BUCKET_NAME,
            "{{ task_instance.xcom_pull(task_ids='generate_production_filename', key='filename') }}"
        ]
    )

    branch_task = BranchPythonOperator(
        task_id='branch_based_on_drift',
        python_callable=branch_based_on_drift,
        provide_context=True,
    )

    ingest_clean_data_task = PythonOperator(
        task_id='ingest_clean_data',
        python_callable=ingest_clean_data,
        op_args=[
            'datasets',
            "{{ task_instance.xcom_pull(task_ids='generate_production_filename', key='filename') }}"
        ]
    )

    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        op_args=[
            "{{ task_instance.xcom_pull(task_ids='ingest_clean_data', key='cleaned_data') }}"
        ]
    )

    skip_training_task = PythonOperator(
        task_id='skip_training',
        python_callable=lambda: print("No drift detected, skipping training."),
    )

    generate_filename_task >> get_and_preprocess_production_data_task >> get_last_used_dataset_task >> detect_drift_task >> branch_task
    branch_task >> ingest_clean_data_task >> train_model_task
    branch_task >> skip_training_task