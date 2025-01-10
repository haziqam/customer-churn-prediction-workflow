import numpy as np
import mlflow

def calculate_psi(expected, actual):
    """
    Calculate the Population Stability Index (PSI) to detect drift.
    
    Args:
        expected (np.array): Expected distribution (training data).
        actual (np.array): Current distribution (new data).
        bins (int): Number of bins for comparison.
    
    Returns:
        float: The PSI value.
    """
    bins = 10
    expected_hist, _ = np.histogram(expected, bins=bins)
    actual_hist, _ = np.histogram(actual, bins=bins)
    expected_perc = expected_hist / sum(expected_hist)
    actual_perc = actual_hist / sum(actual_hist)
    psi = np.sum((expected_perc - actual_perc) * np.log(expected_perc / actual_perc))
    return psi

def log_psi(expected, actual):
    """
    Logs the PSI value to MLflow.
    
    Args:
        expected (np.array): Expected distribution.
        actual (np.array): Current distribution.
    """
    psi_value = calculate_psi(expected, actual)
    with mlflow.start_run(run_name="drift_detection"):
        mlflow.log_metric("psi", psi_value)
        mlflow.log_param("psi_threshold", 0.1)
        mlflow.log_param("drift_detected", psi_value > 0.1)
        print(f"PSI value: {psi_value}")
