from train_model import train_and_log_model
from drift_detection import calculate_psi
import pandas as pd
import numpy as np

def retrain_model_if_drift(training_data_path, current_data_path, model_output_path, psi_threshold=0.1):
    """
    Retrains the model if data drift is detected.

    Args:
        training_data_path (str): Path to the training dataset.
        current_data_path (str): Path to the current dataset.
        model_output_path (str): Path to save the retrained model.
        psi_threshold (float): PSI threshold for drift detection.
    """
    # Load the training and current datasets
    training_data = pd.read_csv(training_data_path)
    current_data = pd.read_csv(current_data_path)

    # Compare the distributions of numerical features
    for column in training_data.select_dtypes(include=np.number).columns:
        expected = training_data[column]
        current = current_data[column]

        # Calculate PSI
        psi = calculate_psi(expected, current)

        print(f"PSI for {column}: {psi:.4f}")
        if psi > psi_threshold:
            print(f"Drift detected in {column} (PSI = {psi:.4f}). Retraining model...")
            train_and_log_model(current_data_path, model_output_path)
            return

    print("No drift detected. Retraining not required.")


if __name__ == "__main__":
    retrain_model_if_drift(
        training_data_path="data/processed_data.csv",
        current_data_path="data/current_data.csv", # live data
        model_output_path="models/retrained_model.pkl",
        psi_threshold=0.1
    )
