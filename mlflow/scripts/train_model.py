from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

def train_and_log_model(data_path, features, model_path="models/random_forest_model.pkl"):
    """
    Trains a Random Forest model and logs to MLflow.
    
    Args:
        data_path (str): Path to the processed data.
        features (list of str): List of columns to use as features for training.
        model_path (str): Path to save the trained model.
    """
    # Load processed data
    df = pd.read_csv(data_path)

    # Drop non-relevant columns (e.g., ID columns)
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)

    # Validate feature columns
    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        raise ValueError(f"The following features are not in the dataset: {missing_features}")

    # Select the specified features
    X = df[features]
    y = df["Churn"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)


    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    model_name = "customer_churn_model"

    # Start MLflow run
    with mlflow.start_run(run_name="model_training"):
        # Train model
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)  # Now compatible with numeric labels

        # Log parameters and metrics
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("features", features)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Log the model and auto-increment version
        print(f"Registering model: {model_name}")
        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="random_forest_model",
                registered_model_name=model_name
            )
            print(f"Model logged and registered as {model_name}")
        except mlflow.exceptions.MlflowException as e:
            print(f"Error registering model: {e}")

# Example usage
if __name__ == "__main__":
    # List the features you want to include for training
    feature_columns = ["gender", "SeniorCitizen", "tenure", "MonthlyCharges"]
    train_and_log_model("data/processed_data.csv", features=feature_columns)
