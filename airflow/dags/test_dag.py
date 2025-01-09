from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id='spark_submit_test',
    schedule_interval=None,
    start_date=datetime(2025, 1, 8),
    catchup=False,
) as dag:

    spark_submit_task = SparkSubmitOperator(
        task_id='submit_spark_job',
        application='/opt/airflow/functions/wordcount.py',
        conn_id='spark_default',
        name='wordcount_job',
        application_args=['/opt/workspace/input.txt', '/opt/workspace/output/'],
        conf={
            'spark.executor.memory': '512m',
            'spark.hadoop.fs.local.block.size': '134217728',
            'spark.hadoop.fs.permissions.umask-mode': '000',
        },
        verbose=True,
    )

    # Define the task sequence
    spark_submit_task