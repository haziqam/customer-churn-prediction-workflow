from train_model import train_and_log_model
from drift_detection import calculate_psi
import pandas as pd
import numpy as np

def retrain_model_if_drift(data_path, psi_threshold=0.1):
    """
    Retrains the model if data drift is detected.
    
    Args:
        data_path (str): Path to the current dataset.
        psi_threshold (float): PSI threshold for drift detection.
    """
    # Simulate expected and current distributions
    expected = np.random.normal(0, 1, 1000)  # Example: Training distribution
    current = np.random.normal(0.5, 1, 1000)  # Example: Current data distribution

    # Calculate PSI
    psi = calculate_psi(expected, current)

    if psi > psi_threshold:
        print("Drift detected! Retraining model...")
        train_and_log_model(data_path, "models/retrained_model.pkl")
    else:
        print("No drift detected. Retraining not required.")

# Example usage
if __name__ == "__main__":
    retrain_model_if_drift("data/processed_data.csv")
