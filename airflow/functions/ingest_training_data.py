from pyspark.sql import SparkSession
import pyspark.pandas as ps
import boto3
import sys

def ingest_training_data(input_path, bucket_name, object_key):
    # Create a Spark session
    spark = SparkSession.builder.appName("Data Ingestion").getOrCreate()

    # Read the dataset into a pandas-on-Spark DataFrame
    df = ps.read_csv(input_path)

    # Save the ingested data to a local file
    temp_file = "/tmp/ingested_data.csv"
    df.to_csv(temp_file, index=False)

    # Upload the file to MinIO
    s3 = boto3.client(
        's3',
        endpoint_url='http://minio:9000',
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin',
    )
    s3.upload_file(temp_file, bucket_name, object_key)
    print(f"Ingested data uploaded to MinIO: {bucket_name}/{object_key}")

    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    input_path = sys.argv[1]
    bucket_name = sys.argv[2]
    object_key = sys.argv[3]
    ingest_training_data(input_path, bucket_name, object_key)
