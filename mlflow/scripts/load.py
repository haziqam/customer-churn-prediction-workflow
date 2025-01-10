import mlflow.sklearn
import pandas as pd

def load_and_predict(model_name, stage, input_data, feature_columns):
    """
    Loads a registered model from MLflow and makes predictions using provided input data.

    Args:
        model_name (str): Name of the registered model in MLflow.
        stage (str): Model stage (e.g., 'Production', 'Staging').
        input_data (pd.DataFrame): Input data as a pandas DataFrame.
        feature_columns (list): List of feature columns to use for predictions.

    Returns:
        list: Predictions from the loaded model.
    """
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Construct the model URI
    model_uri = f"models:/{model_name}/{stage}"
    print(f"Loading model from: {model_uri}")

    # Load the model
    model = mlflow.sklearn.load_model(model_uri)
    print("Model loaded successfully.")

    # Ensure only the specified feature columns are used
    input_data = input_data[feature_columns]

    # Make predictions
    predictions = model.predict(input_data)
    return predictions

# Example usage
if __name__ == "__main__":
    # Specify the feature columns to use
    feature_columns = ["gender", "SeniorCitizen", "tenure", "MonthlyCharges"]

    # Create the input data
    input_data = pd.DataFrame({
        'gender': ['Female', 'Male'],           # Categorical
        'SeniorCitizen': [0, 1],                # Binary: 0 (No), 1 (Yes)
        'tenure': [12, 24],                     # Numerical
        'MonthlyCharges': [70.35, 99.45]        # Numerical
    })

    # Load and predict
    predictions = load_and_predict("customer_churn_model", "None", input_data, feature_columns)
    print("Predictions:", predictions)
