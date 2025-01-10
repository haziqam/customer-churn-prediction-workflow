import pandas as pd
import mlflow

def preprocess_data(input_path, output_path):
    """
    Preprocesses the raw data by handling missing values, 
    encoding categorical variables, and saving the cleaned dataset.
    
    Args:
        input_path (str): Path to the raw data.
        output_path (str): Path to save the processed data.
    """
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Start an MLflow run for preprocessing
    with mlflow.start_run(run_name="data_preprocessing"):
        # Load data
        df = pd.read_csv(input_path)
        
        # Log initial data shape
        mlflow.log_param("initial_data_shape", df.shape)
        
        # Handle missing values
        df.fillna(0, inplace=True)
        mlflow.log_param("missing_value_strategy", "fillna")

        # Save the processed data
        df.to_csv(output_path, index=False)
        
        # Log processed data artifact
        mlflow.log_artifact(output_path, artifact_path="processed_data")
        print(f"Processed data saved to {output_path}")

# Example usage
if __name__ == "__main__":
    preprocess_data("data/data.csv", "data/processed_data.csv")
