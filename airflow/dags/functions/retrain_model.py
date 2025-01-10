from train_model import train_model
from drift_detection import calculate_psi
import pandas as pd
import numpy as np

def retrain_model_if_drift(training_data_path, current_data_path):
    """
    Retrains the model if data drift is detected.

    Args:
        training_data_path (str): Path to the training dataset.
        current_data_path (str): Path to the current dataset.
        model_output_path (str): Path to save the retrained model.
        psi_threshold (float): PSI threshold for drift detection.
    """
    psi_threshold=0.1
    bins=10
    # Load the training and current datasets
    training_data = pd.read_csv(training_data_path)
    current_data = pd.read_csv(current_data_path)

    total_psi = 0
    for column in training_data.select_dtypes(include=np.number).columns:
        expected = training_data[column]
        current = current_data[column]
        psi = calculate_psi(expected, current, bins=bins)
        total_psi += psi
        print(f"PSI for {column}: {psi:.4f}")

    print(f"Total PSI: {total_psi:.4f}")
    if total_psi > psi_threshold:
        print("Overall drift detected! Retraining model...")
        train_model(current_data_path)
    else:
        print("No significant overall drift detected. Retraining not required.")

# if __name__ == "__main__":
#     retrain_model_if_drift(
#         training_data_path="data/processed_data.csv",
#         current_data_path="data/current_data.csv", # live data
#         model_output_path="models/retrained_model.pkl",
#         psi_threshold=0.1,
#         bins=10,
#     )
