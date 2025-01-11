from pyspark.sql import SparkSession
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
from pyspark.ml.feature import StringIndexer
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

    df_spark = spark.read.csv(input_path, header=True, inferSchema=True)
    # Drop duplicates and NaNs
    df_spark = df_spark.dropDuplicates().dropna()

    # Drop unnecessary columns
    df_spark = df_spark.drop("customerID")

    # Map gender column
    df_spark = df_spark.withColumn("gender", when(col("gender") == "Male", 0).otherwise(1))

    # Map binary columns
    binary_col = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
    for col_name in binary_col:
        df_spark = df_spark.withColumn(col_name, when(col(col_name) == "Yes", 1).otherwise(0))

    # Map categorical columns
    categorical_col = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod"
    ]
    for col_name in categorical_col:
        indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_index")
        df_spark = indexer.fit(df_spark).transform(df_spark).drop(col_name).withColumnRenamed(f"{col_name}_index", col_name)

    # Convert TotalCharges to numeric
    df_spark = df_spark.withColumn("TotalCharges", col("TotalCharges").cast("float"))
    df_spark = df_spark.fillna({"TotalCharges": 0.0})


    temp_file = f"/tmp/preprocessed_data_{uuid.uuid4()}.csv"
    df_spark.toPandas().to_csv(temp_file, index=False)

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
    print(f"File uploaded to S3: {bucket_name}/{output_object_key}")
    print("=============================================================")

    spark.stop()

if __name__ == "__main__":
    input_path = sys.argv[1]
    bucket_name = sys.argv[2]
    output_object_key = sys.argv[3]
    preprocess(input_path, bucket_name, output_object_key)