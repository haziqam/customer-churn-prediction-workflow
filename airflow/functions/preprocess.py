from pyspark.sql import SparkSession
import pyspark.pandas as ps
import boto3
import sys

def preprocess(bucket_name, object_key, output_key):
    # Create a Spark session
    spark = SparkSession.builder.appName("Data Preprocessing").getOrCreate()

    # Download the file from MinIO
    temp_file = "/tmp/ingested_data.csv"
    s3 = boto3.client(
        's3',
        endpoint_url='http://minio:9000',
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin',
    )
    s3.download_file(bucket_name, object_key, temp_file)

    # Load the data
    df = ps.read_csv(temp_file)

    # Perform preprocessing
    df = df.drop_duplicates()
    df = df.dropna()

    gender_mapping = {"Male": 0, "Female": 1}
    df["gender"] = df["gender"].map(gender_mapping)

    binary_mapping = {"No": 0, "Yes": 1}
    binary_col = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col_name in binary_col:
        df[col_name] = df[col_name].map(binary_mapping)

    categorical_col = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
    ]
    for col_name in categorical_col:
        df[col_name] = df[col_name].astype("category").cat.codes

    # Save the preprocessed data back to MinIO
    preprocessed_file = "/tmp/preprocessed_data.csv"
    df.to_csv(preprocessed_file, index=False)
    s3.upload_file(preprocessed_file, bucket_name, output_key)
    print(f"Preprocessed data uploaded to MinIO: {bucket_name}/{output_key}")

    spark.stop()

if __name__ == "__main__":
    bucket_name = sys.argv[1]
    object_key = sys.argv[2]
    output_key = sys.argv[3]
    preprocess(bucket_name, object_key, output_key)
