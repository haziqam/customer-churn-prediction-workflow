import pandas as pd
import numpy as np
import sys
import boto3
from io import StringIO

def calculate_psi(expected, actual, bins = 10):
    """
    Calculate the Population Stability Index (PSI) to detect drift.
    
    Args:
        expected (np.array): Expected distribution (training data).
        actual (np.array): Current distribution (new data).
        bins (int): Number of bins for comparison.
    
    Returns:
        float: The PSI value.
    """
    expected_hist, _ = np.histogram(expected, bins=bins)
    actual_hist, _ = np.histogram(actual, bins=bins)
    expected_perc = expected_hist / sum(expected_hist)
    actual_perc = actual_hist / sum(actual_hist)
    psi = np.sum((expected_perc - actual_perc) * np.log(expected_perc / actual_perc))
    return psi

def detect_drift(training_data_path, bucket_name, current_data_path, psi_threshold=0.1, bins=10, **kwargs) -> bool:
    """
    Retrains the model if data drift is detected.

    Args:
        training_data_path (str): Path to the training dataset.
        current_data_path (str): Path to the current dataset.
        model_output_path (str): Path to save the retrained model.
        psi_threshold (float): PSI threshold for drift detection.
    """
    s3 = boto3.client(
        's3',
        endpoint_url='http://minio:9000',
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin'
    )

    # Load the training and current datasets
    current_data_object = s3.get_object(Bucket=bucket_name, Key=current_data_path)
    current_data_csv = current_data_object["Body"].read().decode('utf-8')
    current_data = pd.read_csv(StringIO(current_data_csv))

    training_data_object = s3.get_object(Bucket=bucket_name, Key=training_data_path)
    training_data_csv = training_data_object["Body"].read().decode('utf-8')
    training_data = pd.read_csv(StringIO(training_data_csv))

    print("current_data df")
    print(current_data.describe())
    print(current_data.head())

    print("training_data")
    print(training_data.describe())
    print(training_data.head())

    total_psi = 0
    for column in training_data.select_dtypes(include=np.number).columns:
        expected = training_data[column]
        current = current_data[column]
        print("========EXPECTED============", expected.dtype)
        print("========CURRENT==========", current.dtype)
        psi = calculate_psi(expected, current, bins=bins)
        if np.isnan(psi):
            print(f"PSI calculation produced NaN for column: {column}")
        else:
            total_psi += psi
        print(f"PSI for {column}: {psi:.4f}")

    print(f"Total PSI: {total_psi:.4f}")
    if total_psi > psi_threshold:
        kwargs['ti'].xcom_push(key="drift", value= True)
    else:
        kwargs['ti'].xcom_push(key="drift", value= False)

if __name__ == "__main__":
    original_data = sys.argv[1]
    current_data = sys.argv[2]
    detect_drift(original_data, current_data)
