import pandas as pd
import boto3
from io import StringIO

def ingest_clean_data(bucket_name, object_key, **kwargs):
    print("=============================================================")
    print("bucket_name: ", bucket_name)
    print("object_key: ", object_key)
    print("=============================================================")

    print("=============================================================")
    print("Create S3 client")
    print("=============================================================")

    s3 = boto3.client(
        's3',
        endpoint_url='http://minio:9000',
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin'
    )

    print("=============================================================")
    print("Getting object")
    print("=============================================================")
    
    csv_object = s3.get_object(Bucket=bucket_name, Key=object_key)
    csv_data = csv_object['Body'].read().decode('utf-8')
    
    print("=============================================================")
    print("Read csv")
    print("=============================================================")

    df = pd.read_csv(StringIO(csv_data))

    kwargs['ti'].xcom_push(key='cleaned_data', value=df.to_dict())
    
    print("=============================================================")
    print("Data successfully ingested and placed in XCom.")
    print("=============================================================")
