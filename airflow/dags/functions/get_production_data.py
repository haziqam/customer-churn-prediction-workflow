from pyspark.sql import SparkSession
import pandas as pd
import uuid
import numpy as np
import pyspark.pandas as ps
import boto3
import sys
# from functions.ensure_bucket_exists import ensure_bucket_exists

def get_and_preprocess_production_data(input_path, bucket_name, output_object_key, num_samples=1000):
    production_data = get_production_data(input_path, num_samples)
    df_spark = ps.from_pandas(production_data)
    preprocess_production_data(bucket_name, output_object_key, df_spark)

def get_production_data(input_path, num_samples):
    synthetic_data = pd.DataFrame()
    original_df = pd.read_csv(input_path)

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

    for column in original_df.columns:
        if column == 'customerID':
            synthetic_data[column] = uuid.uuid4()
        elif column in unique_values:
            # Use np.random.choice to randomly sample from the unique values
            synthetic_data[column] = np.random.choice(unique_values[column], num_samples)
        elif column in ['MonthlyCharges', 'TotalCharges']:
            # Generate random decimal numbers with 2 decimal places
            synthetic_data[column] = np.round(np.random.uniform(10, 200, num_samples), 2)
        else:
            # Randomly sample existing categorical values
            synthetic_data[column] = np.random.choice(original_df[column].dropna().unique(), num_samples)
    return synthetic_data


def preprocess_production_data(bucket_name, output_object_key, df_spark):
    spark = SparkSession.builder.appName("Data Preprocessing").getOrCreate()
    df_spark = df_spark.drop_duplicates()
    df_spark = df_spark.dropna()
    df_spark = df_spark.drop(columns=['customerID'])

    gender_mapping = {"Male": 0, "Female": 1}
    df_spark["gender"] = df_spark["gender"].map(gender_mapping)

    binary_mapping = {"No": 0, "Yes": 1}
    binary_col = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
    for col_name in binary_col:
        df_spark[col_name] = df_spark[col_name].map(binary_mapping)

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
        df_spark[col_name] = df_spark[col_name].astype("category").cat.codes

    df_spark['TotalCharges'] = ps.to_numeric(df_spark['TotalCharges'], errors='coerce').fillna(0)

    print("=============================================================")
    print("Done preprocessing")    
    print(df_spark.head())
    print("=============================================================")

    pandas_df = df_spark.to_pandas()
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

    # ensure_bucket_exists(s3, bucket_name)

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
    get_and_preprocess_production_data(input_path, bucket_name, output_object_key)
