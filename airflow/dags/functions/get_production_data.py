from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.types import StringType, IntegerType, FloatType, StructType, StructField
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
from pyspark.ml.feature import StringIndexer
import uuid
import numpy as np
import uuid
import numpy as np
import boto3
import sys
import random

def get_and_preprocess_production_data(bucket_name, output_object_key, num_samples=1000):
    SOURCE_DATASET_FILE = '/opt/workspace/dataset.csv' # the training dataset only used as a reference to generate dummy data as the "production data" 
    df_spark = get_production_data(SOURCE_DATASET_FILE, num_samples)
    preprocess_production_data(bucket_name, output_object_key, df_spark)

def get_production_data(input_path, num_samples):
    # Start a Spark session
    spark = SparkSession.builder.appName("Generate Production Data").getOrCreate()

    # Read schema from reference dataset
    reference_df = spark.read.csv(input_path, header=True, inferSchema=True)
    schema = reference_df.schema

    # Define unique values for categorical columns
    unique_values = {
        'gender': ['Female', 'Male'],
        'SeniorCitizen': [0, 1],
        'Partner': ['Yes', 'No'],
        'Dependents': ['No', 'Yes'],
        'PhoneService': ['No', 'Yes'],
        'MultipleLines': ['No phone service', 'No', 'Yes'],
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'OnlineSecurity': ['No', 'Yes', 'No internet service'],
        'OnlineBackup': ['Yes', 'No', 'No internet service'],
        'DeviceProtection': ['No', 'Yes', 'No internet service'],
        'TechSupport': ['No', 'Yes', 'No internet service'],
        'StreamingTV': ['No', 'Yes', 'No internet service'],
        'StreamingMovies': ['No', 'Yes', 'No internet service'],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'PaperlessBilling': ['Yes', 'No'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        'Churn': ['No', 'Yes'],
    }

    # Generate synthetic data
    synthetic_data = []
    for i in range(num_samples):
        row = {
            'customerID': str(random.randint(100000, 999999)),
            'MonthlyCharges': float(np.round(np.random.uniform(0, 1000000), 2)),
            'TotalCharges': float(np.round(np.random.uniform(0, 1000000), 2)),
            'tenure': random.randint(1, 100),
            **{
                col: (
                    np.random.choice(unique_values[col]).item() if col in unique_values else None
                )
                for col in schema.fieldNames() if col not in ['customerID', 'MonthlyCharges', 'TotalCharges', 'tenure']
            },
        }
        synthetic_data.append(row)

    # Define PySpark schema explicitly
    explicit_schema = StructType([
        StructField("customerID", StringType(), True),
        StructField("gender", StringType(), True),
        StructField("SeniorCitizen", IntegerType(), True),
        StructField("Partner", StringType(), True),
        StructField("Dependents", StringType(), True),
        StructField("PhoneService", StringType(), True),
        StructField("MultipleLines", StringType(), True),
        StructField("InternetService", StringType(), True),
        StructField("OnlineSecurity", StringType(), True),
        StructField("OnlineBackup", StringType(), True),
        StructField("DeviceProtection", StringType(), True),
        StructField("TechSupport", StringType(), True),
        StructField("StreamingTV", StringType(), True),
        StructField("StreamingMovies", StringType(), True),
        StructField("Contract", StringType(), True),
        StructField("PaperlessBilling", StringType(), True),
        StructField("PaymentMethod", StringType(), True),
        StructField("Churn", StringType(), True),
        StructField("MonthlyCharges", FloatType(), True),
        StructField("TotalCharges", FloatType(), True),
        StructField("tenure", IntegerType(), True)
    ])

    # Create PySpark DataFrame
    synthetic_df = spark.createDataFrame(synthetic_data, schema=explicit_schema)

    print("=============================================================")
    print("Done generating data")
    synthetic_df.show(5)
    synthetic_df.describe().show()
    print("=============================================================")

    return synthetic_df

def preprocess_production_data(bucket_name, output_object_key, df_spark):
    spark = SparkSession.builder.appName("Data Preprocessing").getOrCreate()

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

    # Write to CSV and upload to S3
    temp_file = f"/tmp/preprocessed_data_{uuid.uuid4()}.csv"
    df_spark.toPandas().to_csv(temp_file, index=False)

    # Upload to S3
    s3 = boto3.client(
        "s3",
        endpoint_url="http://minio:9000",
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
    )
    s3.upload_file(temp_file, bucket_name, output_object_key)

    print("=============================================================")
    print(f"File uploaded to S3: {bucket_name}/{output_object_key}")
    print("=============================================================")

    spark.stop()

if __name__ == "__main__":
    bucket_name = sys.argv[1]
    output_object_key = sys.argv[2]
    get_and_preprocess_production_data(bucket_name, output_object_key)
