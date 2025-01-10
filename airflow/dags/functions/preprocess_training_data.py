from pyspark.sql import SparkSession
import pyspark.pandas as ps
import uuid
import boto3
import sys
from ensure_bucket_exists import ensure_bucket_exists

def preprocess(input_path, bucket_name, output_object_key):
    print("=============================================================")
    print("input_path: ", input_path)
    print("bucket_name: ", bucket_name)
    print("output_object_key: ", output_object_key)
    print("=============================================================")
    
    spark = SparkSession.builder.appName("Data Preprocessing").getOrCreate()


    df = ps.read_csv(input_path)
    print("=============================================================")
    print("Read csv")
    print("=============================================================")


    df = df.drop_duplicates()
    df = df.dropna()
    df = df.drop(columns=['customerID'])

    gender_mapping = {"Male": 0, "Female": 1}
    df["gender"] = df["gender"].map(gender_mapping)

    binary_mapping = {"No": 0, "Yes": 1}
    binary_col = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
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

    df['TotalCharges'] = ps.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    print("=============================================================")
    print("Done preprocessing")    
    print(df.head())
    print("=============================================================")

    pandas_df = df.to_pandas()
    temp_file = f"/tmp/preprocessed_data_{uuid.uuid4()}.csv"
    pandas_df.to_csv(temp_file, index=False)

    print("=============================================================")
    print("Create S3 client")
    print("=============================================================")

    s3 = boto3.client(
        's3',
        endpoint_url='http://minio:9000',
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin',
    )

    ensure_bucket_exists(s3, bucket_name)

    print("=============================================================")
    print("Uploading file to S3")
    print("=============================================================")

    s3.upload_file(temp_file, bucket_name, output_object_key)

    print("=============================================================")
    print("File uploaded to S3")
    print("=============================================================")

    spark.stop()

if __name__ == "__main__":
    input_path = sys.argv[1]
    bucket_name = sys.argv[2]
    output_object_key = sys.argv[3]
    preprocess(input_path, bucket_name, output_object_key)