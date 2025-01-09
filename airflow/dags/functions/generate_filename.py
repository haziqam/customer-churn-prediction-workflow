from datetime import datetime

def generate_filename(**context):
    timestamp = datetime.now()
    filename = f'{timestamp.isoformat()}.csv'
    context['task_instance'].xcom_push(key='filename', value=filename)
    return filename