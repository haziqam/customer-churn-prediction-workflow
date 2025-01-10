import pandas as pd
import uuid
import numpy as np
from faker import Faker
import boto3
import sys



# Function to simulate new data
def generate_synthetic_data(input_path,bukcet_name, output_object_key, num_samples=1000):
    fake = Faker()

    synthetic_data = pd.DataFrame()
    original_df = pd.read_csv(input_path)

    for column in original_df.columns:
        if column == 'customerID':
            # Generate unique customer IDs
            continue
        elif column in ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']:
            # Generate binary data (0 or 1)
            synthetic_data[column] = np.random.randint(0, 2, num_samples)
        elif column in ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract']:
            # Generate categorical data (0, 1, or 2)
            synthetic_data[column] = np.random.randint(0, 3, num_samples)
        elif column in ['MonthlyCharges', 'TotalCharges']:
            # Generate random decimal numbers with 2 decimal places
            synthetic_data[column] = np.round(np.random.uniform(10, 200, num_samples), 2)
        elif column in ['PaymentMethod']:
            synthetic_data[column] = np.random.randint(0, 5, num_samples)
        else:
            # Randomly sample existing categorical values
            synthetic_data[column] = np.random.choice(original_df[column].dropna().unique(), num_samples)
    
    s3 = boto3.client(
        's3',
        endpoint_url='http://minio:9000',
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin',
    ) 
    temp_file = f"/tmp/production_data_{uuid.uuid4()}.csv"
    synthetic_data.to_csv(temp_file, index=False)

    s3.upload_file(temp_file, bukcet_name, output_object_key)

if __name__ == "__main__":
    input_path = sys.argv[1]
    df = pd.read_csv(input_path)
    bucket_name = sys.argv[2]
    output_object_key = sys.argv[3]
    generate_synthetic_data(df, bucket_name, output_object_key)
