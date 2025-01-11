import boto3

def get_last_used_dataest(**kwargs):
    s3 = boto3.client(
        's3',
        endpoint_url='http://minio:9000',
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin'
    )

    bucket_name = 'datasets'
    last_used_dataset_obj = s3.get_object(Bucket=bucket_name, Key='last_used_dataset.txt')
    last_used_dataset_key = last_used_dataset_obj['Body'].read().decode('utf-8').strip()

    print("=============================================================")
    print("last used dataset is: ", last_used_dataset_key)
    print("=============================================================")
    
    kwargs['ti'].xcom_push(key='last_used_dataset', value=last_used_dataset_key)