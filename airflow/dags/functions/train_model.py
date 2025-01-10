import pandas as pd
import ast
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

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

    # Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model_name = "customer_churn_rf_model"

    # Logistic Regression Classifier
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model_name = "customer_churn_lr_model"

    mlflow.set_tracking_uri("http://mlflow:5000")

    # Train and log Random Forest model
    with mlflow.start_run(run_name="rf_model_training"):
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)

        rf_acc = accuracy_score(y_test, y_pred_rf)
        rf_f1 = f1_score(y_test, y_pred_rf)

        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", rf_acc)
        mlflow.log_metric("f1_score", rf_f1)

        try:
            mlflow.sklearn.log_model(
                sk_model=rf_model,
                artifact_path="random_forest_model",
                registered_model_name=rf_model_name
            )
            print(f"Random Forest model logged and registered as {rf_model_name}")
        except mlflow.exceptions.MlflowException as e:
            print(f"Error registering Random Forest model: {e}")

    # Train and log Logistic Regression model
    with mlflow.start_run(run_name="lr_model_training"):
        lr_model.fit(X_train, y_train)
        y_pred_lr = lr_model.predict(X_test)

        lr_acc = accuracy_score(y_test, y_pred_lr)
        lr_f1 = f1_score(y_test, y_pred_lr)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_metric("accuracy", lr_acc)
        mlflow.log_metric("f1_score", lr_f1)

        try:
            mlflow.sklearn.log_model(
                sk_model=lr_model,
                artifact_path="logistic_regression_model",
                registered_model_name=lr_model_name
            )
            print(f"Logistic Regression model logged and registered as {lr_model_name}")
        except mlflow.exceptions.MlflowException as e:
            print(f"Error registering Logistic Regression model: {e}")
