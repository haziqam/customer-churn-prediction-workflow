import pandas as pd
import ast
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import os

def train_model(df_dict_str):
    df_dict = ast.literal_eval(df_dict_str)
    df = pd.DataFrame(df_dict)

    print("=============================================================")
    print(df.head())
    print("=============================================================")

    print("=============================================================")
    print("Train models")
    print("=============================================================")

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://minio:9000"
    mlflow.set_tracking_uri("http://mlflow:5000")
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