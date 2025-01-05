from airflow.decorators import dag, task
from pendulum import datetime

@dag(
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
)
def minimal_test_dag():
    
    @task()
    def say_hello():
        print("Hello")
        return "Hello"
    
    say_hello()

minimal_test_dag()