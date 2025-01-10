import numpy as np
import mlflow

def calculate_psi(expected, actual):
    """
    Calculate the Population Stability Index (PSI) to detect drift.
    
    Args:
        expected (np.array): Expected distribution (training data).
        actual (np.array): Current distribution (new data).
        bins (int, optional): Number of bins for comparison. If None, Doane's formula will be used.
    
    Returns:
        float: The PSI value.
    """
    # Combine expected and actual datasets to determine bin edges
    full_dataset = np.concatenate((expected, actual))

    # Determine bin edges
    bin_edges = np.linspace(min(min(expected), min(actual)), max(max(expected), max(actual)),  10)

    # Calculate histograms for expected and actual distributions
    expected_hist, _ = np.histogram(expected, bins=bin_edges)
    actual_hist, _ = np.histogram(actual, bins=bin_edges)

    # Convert counts to proportions
    expected_proportions = expected_hist / np.sum(expected_hist)
    actual_proportions = actual_hist / np.sum(actual_hist)

    # Replace zero proportions to avoid division by zero or log of zero errors
    expected_proportions = np.where(expected_proportions == 0, 1e-6, expected_proportions)
    actual_proportions = np.where(actual_proportions == 0, 1e-6, actual_proportions)

    # Calculate PSI
    psi_values = (actual_proportions - expected_proportions) * np.log(actual_proportions / expected_proportions)
    psi = np.sum(psi_values)

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
