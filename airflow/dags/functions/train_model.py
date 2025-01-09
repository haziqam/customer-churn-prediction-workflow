import pandas as pd
import ast
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
import mlflow.sklearn

def train_model(df_dict_str):
    df_dict = ast.literal_eval(df_dict_str)
    df = pd.DataFrame(df_dict)

    print("=============================================================")
    print(df.head())
    print("=============================================================")

    print("=============================================================")
    print("Train model")
    print("=============================================================")

      # Drop non-relevant columns (e.g., ID columns)
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)

    # Select the specified features
    # TODO: nanti kalo ada var features
    # X = df[features] 
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Encode the target variable
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)  # Converts 'Yes'/'No' to 1/0

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=["object"]).columns

    # Create a preprocessor for categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_cols)
        ],
        remainder="passthrough"  # Keep numerical columns as is
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define the pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    model_name = "customer_churn_model"
    # Start MLflow run
    with mlflow.start_run(run_name="model_training"):
        # Train model
        pipeline.fit(X_train, y_train)

        # Evaluate model
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)  

        # Log parameters and metrics
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        # mlflow.log_param("features", features) 
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Log the model and auto-increment version
        print(f"Registering model: {model_name}")
        try:
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="random_forest_model",
                registered_model_name=model_name
            )
            print(f"Model logged and registered as {model_name}")
        except mlflow.exceptions.MlflowException as e:
            print(f"Error registering model: {e}")
