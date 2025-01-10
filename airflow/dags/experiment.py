from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from functions.ingest_clean_data import ingest_clean_data
from functions.generate_filename import generate_filename
from functions.train_model import train_model

SOURCE_DATASET_FILE = '/opt/workspace/dataset.csv'
DATASET_BUCKET_NAME = 'datasets'

with DAG(
    dag_id='experiment_dag',
    schedule_interval=None,
    start_date=datetime(2025, 1, 8),
    catchup=False,
) as dag:
    generate_filename_task = PythonOperator(
        task_id='generate_filename',
        python_callable=generate_filename,
        dag=dag
    )

    preprocess_task = SparkSubmitOperator(
        task_id='preprocess_training_data',
        application='/opt/airflow/dags/functions/preprocess_training_data.py',
        conn_id='spark_default',
        name='preprocess_job',
        application_args=[
            SOURCE_DATASET_FILE,
            DATASET_BUCKET_NAME,
            "{{ task_instance.xcom_pull(task_ids='generate_filename', key='filename') }}"
        ],
        conf={
            'spark.executor.memory': '512m',
        },
        verbose=True,
    )

    ingest_clean_data_task = PythonOperator(
        task_id='ingest_clean_data',
        python_callable=ingest_clean_data,
        op_args=[
            'datasets',
            # '2025-01-10T14:51:11.372793.csv'
            "{{ task_instance.xcom_pull(task_ids='generate_filename', key='filename') }}"    
        ],
        provide_context=True,
        dag=dag
    )

    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        op_args=[
            "{{ task_instance.xcom_pull(task_ids='ingest_clean_data', key='cleaned_data') }}"
        ],
        provide_context=True,
        dag=dag
    )

    generate_filename_task >> preprocess_task >> ingest_clean_data_task >> train_model_task

