import pandas as pd
import ast
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import joblib
import boto3
import os
import mlflow
from datetime import datetime

def train_model(df_dict_str):
    df_dict = ast.literal_eval(df_dict_str)
    df = pd.DataFrame(df_dict)

    print("=============================================================")
    print(df.head())
    print("=============================================================")

    print("=============================================================")
    print("Train models")
    print("=============================================================")

    s3 = boto3.client(
        's3',
        endpoint_url='http://minio:9000',
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin'
    )

    timestamp = datetime.now().isoformat()

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model_name = "customer_churn_rf_model"

    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model_name = "customer_churn_lr_model"

    mlflow.set_tracking_uri("http://mlflow:5000")

    # Track the best model
    best_accuracy = -1
    best_f1 = -1
    best_artifact_path = ""

    with mlflow.start_run(run_name="rf_model_training"):
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)

        rf_acc = accuracy_score(y_test, y_pred_rf)
        rf_f1 = f1_score(y_test, y_pred_rf)

        rf_artifact_path = f"rf_model_{timestamp}.pkl"
        mlflow.sklearn.log_model(
            sk_model=rf_model,
            artifact_path=rf_artifact_path,
            registered_model_name=rf_model_name
        )
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Model is uploaded to mlflow-artifacts in S3 (normally this is done automatically by 
        # mlflow.sklearn.log_model(), but it didn't work)
        joblib.dump(rf_model, rf_artifact_path) 
        s3.upload_file(rf_artifact_path, 'mlflow-artifacts', rf_artifact_path)
        
        print("=============================================================")
        print(f"Random Forest model uploaded to S3 mlflow-artifacts/{rf_artifact_path}")
        print("=============================================================")

        os.remove(rf_artifact_path)

        if rf_acc > best_accuracy or (rf_acc == best_accuracy and rf_f1 > best_f1):
            best_accuracy = rf_acc
            best_f1 = rf_f1
            best_artifact_path = rf_artifact_path

    with mlflow.start_run(run_name="lr_model_training"):
        lr_model.fit(X_train, y_train)
        y_pred_lr = lr_model.predict(X_test)

        lr_acc = accuracy_score(y_test, y_pred_lr)
        lr_f1 = f1_score(y_test, y_pred_lr)

        lr_artifact_path = f"lr_model_{timestamp}.pkl"
        mlflow.sklearn.log_model(
            sk_model=lr_model,
            artifact_path=lr_artifact_path,
            registered_model_name=lr_model_name
        )
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_metric("accuracy", lr_acc)
        mlflow.log_metric("f1_score", lr_f1)

        # Model is uploaded to mlflow-artifacts in S3 (normally this is done automatically by 
        # mlflow.sklearn.log_model(), but it didn't work)
        joblib.dump(lr_model, lr_artifact_path) 
        s3.upload_file(lr_artifact_path, 'mlflow-artifacts', lr_artifact_path)
        
        print("=============================================================")
        print(f"Logistic Regression model uploaded to S3 mlflow-artifacts/{lr_artifact_path}")
        print("=============================================================")

        os.remove(lr_artifact_path)

        if lr_acc > best_accuracy or (lr_acc == best_accuracy and lr_f1 > best_f1):
            best_accuracy = lr_acc
            best_f1 = lr_f1
            best_artifact_path = lr_artifact_path

    print("=============================================================")
    print(f"Best model marked in S3 mlflow-artifacts/{lr_artifact_path}")
    print("=============================================================")

    text_content = best_artifact_path
    s3.put_object(
        Bucket='mlflow-artifacts',
        Key='last_used_artifact_path.txt',
        Body=text_content.encode('utf-8')
    )

    
